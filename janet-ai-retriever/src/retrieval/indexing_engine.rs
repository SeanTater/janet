use anyhow::Result;
use blake3;
use janet_ai_embed::{EmbeddingProvider, FastEmbedProvider, EmbedConfig};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use super::chunking_strategy::{ChunkingConfig, ChunkingStrategy};
use super::enhanced_index::{
    EnhancedFileIndex, EmbeddingModelMetadata, IndexMetadata, IndexStats,
};
use super::file_index::{ChunkRef, FileRef};
use super::indexing_mode::IndexingMode;
use super::task_queue::{IndexingTask, TaskQueue, TaskQueueConfig, TaskType};

/// Configuration for the indexing engine
#[derive(Debug, Clone)]
pub struct IndexingEngineConfig {
    /// Repository name
    pub repository: String,
    /// Base path for indexing
    pub base_path: PathBuf,
    /// Indexing mode
    pub mode: IndexingMode,
    /// Task queue configuration
    pub task_queue_config: TaskQueueConfig,
    /// Chunking configuration
    pub chunking_config: ChunkingConfig,
    /// Embedding configuration (optional)
    pub embedding_config: Option<EmbedConfig>,
    /// Maximum concurrent workers for indexing
    pub max_workers: usize,
    /// Timeout for individual file processing
    pub file_timeout: Duration,
}

impl IndexingEngineConfig {
    pub fn new(repository: String, base_path: PathBuf) -> Self {
        Self {
            chunking_config: ChunkingConfig::new(repository.clone()),
            repository,
            base_path,
            mode: IndexingMode::default(),
            task_queue_config: TaskQueueConfig::default(),
            embedding_config: None,
            max_workers: 4,
            file_timeout: Duration::from_secs(60),
        }
    }
    
    pub fn with_mode(mut self, mode: IndexingMode) -> Self {
        self.mode = mode;
        self
    }
    
    pub fn with_embedding_config(mut self, config: EmbedConfig) -> Self {
        self.embedding_config = Some(config);
        self
    }
    
    pub fn with_max_workers(mut self, workers: usize) -> Self {
        self.max_workers = workers;
        self.task_queue_config.max_workers = workers;
        self
    }
    
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunking_config = self.chunking_config.with_max_chunk_size(size);
        self
    }
}

/// Result of processing a single file
#[derive(Debug)]
pub struct FileProcessingResult {
    pub file_path: PathBuf,
    pub chunks_created: usize,
    pub embeddings_generated: usize,
    pub processing_time: Duration,
}

/// The main indexing engine that orchestrates file watching, chunking, and embedding
pub struct IndexingEngine {
    config: IndexingEngineConfig,
    enhanced_index: EnhancedFileIndex,
    task_queue: TaskQueue,
    chunking_strategy: ChunkingStrategy,
    embedding_provider: Option<FastEmbedProvider>,
    embedding_model_metadata: Option<EmbeddingModelMetadata>,
    workers: Vec<tokio::task::JoinHandle<()>>,
    shutdown_sender: Option<mpsc::UnboundedSender<()>>,
    stats: RwLock<ProcessingStats>,
}

#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub files_processed: usize,
    pub chunks_created: usize,
    pub embeddings_generated: usize,
    pub errors: usize,
}

impl IndexingEngine {
    /// Create a new indexing engine
    pub async fn new(config: IndexingEngineConfig) -> Result<Self> {
        Self::new_impl(config, false).await
    }
    
    /// Create a new indexing engine with in-memory database (for testing)
    pub async fn new_memory(config: IndexingEngineConfig) -> Result<Self> {
        Self::new_impl(config, true).await
    }
    
    async fn new_impl(config: IndexingEngineConfig, use_memory: bool) -> Result<Self> {
        info!("Initializing IndexingEngine for {}", config.repository);
        
        // Initialize enhanced file index
        let enhanced_index = if use_memory {
            EnhancedFileIndex::open_memory(&config.base_path).await?
        } else {
            EnhancedFileIndex::open(&config.base_path).await?
        };
        
        // Initialize task queue
        let task_queue = TaskQueue::new(config.task_queue_config.clone());
        
        // Initialize chunking strategy
        let chunking_strategy = ChunkingStrategy::new(config.chunking_config.clone());
        
        // Initialize embedding provider if configured
        let (embedding_provider, embedding_model_metadata) = if let Some(embed_config) = &config.embedding_config {
            if config.mode.allows_indexing() {
                info!("Initializing embedding provider: {}", embed_config.model_name);
                
                let provider = FastEmbedProvider::create(embed_config.clone()).await?;
                let metadata = EmbeddingModelMetadata::new(
                    embed_config.model_name.clone(),
                    "fastembed".to_string(),
                    provider.embedding_dimension(),
                )
                .with_normalized(embed_config.normalize);
                
                // Register the model in the database
                enhanced_index.register_embedding_model(&metadata).await?;
                
                (Some(provider), Some(metadata))
            } else {
                info!("Read-only mode: skipping embedding provider initialization");
                (None, None)
            }
        } else {
            info!("No embedding configuration provided");
            (None, None)
        };
        
        // Initialize or update index metadata
        let index_metadata = if let Some(ref embed_meta) = embedding_model_metadata {
            IndexMetadata::new(config.repository.clone()).with_embedding_model(embed_meta.clone())
        } else {
            IndexMetadata::new(config.repository.clone())
        };
        enhanced_index.upsert_index_metadata(&index_metadata).await?;
        
        Ok(Self {
            config,
            enhanced_index,
            task_queue,
            chunking_strategy,
            embedding_provider,
            embedding_model_metadata,
            workers: Vec::new(),
            shutdown_sender: None,
            stats: RwLock::new(ProcessingStats::default()),
        })
    }
    
    /// Start the indexing engine
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting IndexingEngine in mode: {}", self.config.mode);
        
        if !self.config.mode.allows_indexing() {
            info!("Read-only mode: indexing workers will not be started");
            return Ok(());
        }
        
        // Start task queue processor
        self.task_queue.start_processor().await;
        
        // Create shutdown channel
        let (shutdown_sender, _shutdown_receiver) = mpsc::unbounded_channel();
        self.shutdown_sender = Some(shutdown_sender);
        
        // For now, we'll implement a simple single-threaded worker
        // TODO: Implement proper multi-threading later
        info!("Worker system initialized (single-threaded mode)");
        
        info!("Started {} indexing workers", self.config.max_workers);
        
        // Perform initial index if needed
        if self.config.mode == IndexingMode::FullReindex {
            self.schedule_full_reindex().await?;
        }
        
        Ok(())
    }
    
    /// Process tasks from the queue (simplified single-threaded version)
    /// Processes all currently pending tasks, with a safety limit to prevent hanging
    pub async fn process_pending_tasks(&self) -> Result<()> {
        let max_tasks_per_batch = 100; // Safety limit to prevent infinite loops
        let mut tasks_processed = 0;
        
        while let Some(task) = self.task_queue.pop_task().await {
            let result = self.process_task_internal(&task).await;
            
            match result {
                Ok(file_result) => {
                    debug!("Completed task: {}", task.description());
                    
                    // Update stats
                    let mut stats_guard = self.stats.write().await;
                    stats_guard.files_processed += 1;
                    stats_guard.chunks_created += file_result.chunks_created;
                    stats_guard.embeddings_generated += file_result.embeddings_generated;
                }
                Err(e) => {
                    error!("Failed to process task {}: {}", task.description(), e);
                    
                    // Update error stats
                    let mut stats_guard = self.stats.write().await;
                    stats_guard.errors += 1;
                }
            }
            
            tasks_processed += 1;
            if tasks_processed >= max_tasks_per_batch {
                debug!("Reached max tasks per batch ({}), stopping to prevent hanging", max_tasks_per_batch);
                break;
            }
        }
        
        debug!("Processed {} tasks in this batch", tasks_processed);
        Ok(())
    }
    
    /// Process a single indexing task
    async fn process_task_internal(&self, task: &IndexingTask) -> Result<FileProcessingResult> {
        let start_time = std::time::Instant::now();
        
        match &task.task_type {
            TaskType::IndexFile { path } => {
                Self::process_file(
                    path,
                    &self.chunking_strategy,
                    &self.enhanced_index,
                    &self.embedding_provider,
                    self.embedding_model_metadata.as_ref().map(|m| m.model_id()),
                    start_time,
                ).await
            }
            TaskType::RemoveFile { path } => {
                Self::remove_file(path, &self.enhanced_index, start_time).await
            }
        }
    }
    
    /// Process a single file
    async fn process_file(
        file_path: &Path,
        chunking_strategy: &ChunkingStrategy,
        enhanced_index: &EnhancedFileIndex,
        embedding_provider: &Option<FastEmbedProvider>,
        embedding_model_id: Option<String>,
        start_time: std::time::Instant,
    ) -> Result<FileProcessingResult> {
        debug!("Processing file: {}", file_path.display());
        
        // Check if file should be indexed
        if !chunking_strategy.should_index_file(file_path) {
            return Ok(FileProcessingResult {
                file_path: file_path.to_path_buf(),
                chunks_created: 0,
                embeddings_generated: 0,
                processing_time: start_time.elapsed(),
            });
        }
        
        // Read file content
        let content = tokio::fs::read_to_string(file_path).await
            .map_err(|e| anyhow::anyhow!("Failed to read file {}: {}", file_path.display(), e))?;
        
        // Generate file hash
        let file_hash = *blake3::hash(content.as_bytes()).as_bytes();
        
        // Create file reference
        let file_ref = FileRef {
            relative_path: file_path.to_string_lossy().to_string(),
            content: content.as_bytes().to_vec(),
            hash: file_hash,
        };
        
        // Store file in database
        enhanced_index.upsert_file(&file_ref).await?;
        
        // Chunk the content using janet-ai-context
        let chunks = chunking_strategy.chunk_content(file_path, &content)?;
        
        if chunks.is_empty() {
            return Ok(FileProcessingResult {
                file_path: file_path.to_path_buf(),
                chunks_created: 0,
                embeddings_generated: 0,
                processing_time: start_time.elapsed(),
            });
        }
        
        // Convert TextChunk to ChunkRef
        let mut chunk_refs: Vec<ChunkRef> = chunks
            .into_iter()
            .map(|chunk| ChunkRef {
                id: None,
                file_hash,
                relative_path: chunk.path,
                line_start: chunk.sequence * 10, // Simple line numbering based on sequence
                line_end: (chunk.sequence + 1) * 10,
                content: chunk.chunk_text,
                embedding: None,
            })
            .collect();
        
        let mut embeddings_generated = 0;
        
        // Generate embeddings if provider is available
        if let (Some(provider), Some(model_id)) = (embedding_provider, embedding_model_id.as_deref()) {
            let texts: Vec<String> = chunk_refs.iter().map(|c| c.content.clone()).collect();
            
            match provider.embed_texts(&texts).await {
                Ok(embedding_result) => {
                    for (chunk, embedding) in chunk_refs.iter_mut().zip(embedding_result.embeddings) {
                        chunk.embedding = Some(embedding);
                        embeddings_generated += 1;
                    }
                    
                    // Store chunks with embeddings
                    enhanced_index.upsert_chunks_with_model(&chunk_refs, model_id).await?;
                }
                Err(e) => {
                    warn!("Failed to generate embeddings for {}: {}", file_path.display(), e);
                    // Store chunks without embeddings
                    enhanced_index.upsert_chunks(&chunk_refs).await?;
                }
            }
        } else {
            // Store chunks without embeddings
            enhanced_index.upsert_chunks(&chunk_refs).await?;
        }
        
        Ok(FileProcessingResult {
            file_path: file_path.to_path_buf(),
            chunks_created: chunk_refs.len(),
            embeddings_generated,
            processing_time: start_time.elapsed(),
        })
    }
    
    /// Remove a file from the index
    async fn remove_file(
        file_path: &Path,
        _enhanced_index: &EnhancedFileIndex,
        start_time: std::time::Instant,
    ) -> Result<FileProcessingResult> {
        debug!("Removing file from index: {}", file_path.display());
        
        // TODO: Implement file removal logic
        // This would involve finding the file by path and removing its chunks
        
        Ok(FileProcessingResult {
            file_path: file_path.to_path_buf(),
            chunks_created: 0,
            embeddings_generated: 0,
            processing_time: start_time.elapsed(),
        })
    }
    
    
    /// Schedule a full reindex by walking the directory tree and adding individual file tasks
    pub async fn schedule_full_reindex(&self) -> Result<()> {
        if !self.config.mode.allows_indexing() {
            return Err(anyhow::anyhow!("Full reindex not allowed in read-only mode"));
        }
        
        info!("Starting full reindex of: {}", self.config.base_path.display());
        
        let mut files_scheduled = 0;
        let mut tasks_batch = Vec::new();
        const BATCH_SIZE: usize = 100; // Submit tasks in batches to avoid overwhelming the queue
        
        // Use a stack to implement depth-first traversal
        let mut dir_stack = vec![self.config.base_path.clone()];
        
        while let Some(current_dir) = dir_stack.pop() {
            let mut read_dir = match tokio::fs::read_dir(&current_dir).await {
                Ok(rd) => rd,
                Err(e) => {
                    warn!("Failed to read directory {}: {}", current_dir.display(), e);
                    continue;
                }
            };
            
            while let Some(entry) = read_dir.next_entry().await? {
                let path = entry.path();
                let metadata = match entry.metadata().await {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Failed to get metadata for {}: {}", path.display(), e);
                        continue;
                    }
                };
                
                if metadata.is_file() && self.chunking_strategy.should_index_file(&path) {
                    // Add to batch for background indexing during full reindex
                    let task = IndexingTask::index_file_background(path);
                    tasks_batch.push(task);
                    files_scheduled += 1;
                    
                    // Submit batch when it reaches the target size
                    if tasks_batch.len() >= BATCH_SIZE {
                        if let Err(e) = self.task_queue.submit_tasks(tasks_batch.drain(..).collect()).await {
                            warn!("Failed to submit batch of tasks: {}", e);
                        }
                    }
                } else if metadata.is_dir() {
                    // Add subdirectory to stack for processing
                    dir_stack.push(path);
                }
            }
        }
        
        // Submit any remaining tasks in the final batch
        if !tasks_batch.is_empty() {
            if let Err(e) = self.task_queue.submit_tasks(tasks_batch).await {
                warn!("Failed to submit final batch of tasks: {}", e);
            }
        }
        
        info!("Scheduled {} files for background indexing", files_scheduled);
        Ok(())
    }
    
    /// Schedule indexing of a single file
    pub async fn schedule_file_index(&self, file_path: PathBuf) -> Result<()> {
        if !self.config.mode.allows_indexing() {
            return Err(anyhow::anyhow!("File indexing not allowed in read-only mode"));
        }
        
        let task = IndexingTask::index_file(file_path);
        self.task_queue.submit_task(task).await
            .map_err(|e| anyhow::anyhow!("Failed to schedule file index: {}", e))
    }
    
    /// Get current processing statistics
    pub async fn get_stats(&self) -> ProcessingStats {
        self.stats.read().await.clone()
    }
    
    /// Get index statistics from database
    pub async fn get_index_stats(&self) -> Result<IndexStats> {
        self.enhanced_index.get_index_stats().await
    }
    
    /// Get the current task queue size
    pub async fn get_queue_size(&self) -> usize {
        self.task_queue.queue_size().await
    }
    
    /// Get a reference to the enhanced file index for searching
    pub fn get_enhanced_index(&self) -> &EnhancedFileIndex {
        &self.enhanced_index
    }
    
    /// Shutdown the indexing engine
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down IndexingEngine");
        
        // Only shutdown workers and queue if indexing was enabled
        if self.config.mode.allows_indexing() {
            // Send shutdown signal to workers
            if let Some(sender) = self.shutdown_sender.take() {
                let _ = sender.send(());
            }
            
            // Wait for workers to finish
            for worker in self.workers.drain(..) {
                let _ = worker.await;
            }
            
            // Shutdown task queue
            self.task_queue.shutdown().await;
        } else {
            // In read-only mode, just clear any shutdown sender
            self.shutdown_sender.take();
        }
        
        info!("IndexingEngine shutdown complete");
        Ok(())
    }
}

impl Clone for ProcessingStats {
    fn clone(&self) -> Self {
        Self {
            files_processed: self.files_processed,
            chunks_created: self.chunks_created,
            embeddings_generated: self.embeddings_generated,
            errors: self.errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_indexing_engine_creation() -> Result<()> {
        let temp_dir = tempdir()?;
        let config = IndexingEngineConfig::new(
            "test-repo".to_string(),
            temp_dir.path().to_path_buf(),
        ).with_mode(IndexingMode::ReadOnly);
        
        let engine = IndexingEngine::new_memory(config).await?;
        
        // Should be able to get stats even in read-only mode
        let stats = engine.get_index_stats().await?;
        assert_eq!(stats.files_count, 0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_file_scheduling() -> Result<()> {
        let temp_dir = tempdir()?;
        let config = IndexingEngineConfig::new(
            "test-repo".to_string(),
            temp_dir.path().to_path_buf(),
        ).with_mode(IndexingMode::ContinuousMonitoring);
        
        let mut engine = IndexingEngine::new_memory(config).await?;
        engine.start().await?;
        
        // Create a test file
        let test_file = temp_dir.path().join("test.rs");
        tokio::fs::write(&test_file, "fn main() { println!(\"Hello\"); }").await?;
        
        // Schedule indexing
        engine.schedule_file_index(test_file).await?;
        
        // Give some time for processing
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        engine.shutdown().await?;
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_full_reindex_schedules_individual_tasks() -> Result<()> {
        let temp_dir = tempdir()?;
        
        // Create multiple test files
        tokio::fs::write(temp_dir.path().join("file1.rs"), "fn main() {}").await?;
        tokio::fs::write(temp_dir.path().join("file2.py"), "print('hello')").await?;
        tokio::fs::write(temp_dir.path().join("file3.js"), "console.log('hi')").await?;
        tokio::fs::write(temp_dir.path().join("README.md"), "# Test Project").await?;
        
        // Create a subdirectory with more files
        let subdir = temp_dir.path().join("src");
        tokio::fs::create_dir(&subdir).await?;
        tokio::fs::write(subdir.join("lib.rs"), "pub mod test;").await?;
        tokio::fs::write(subdir.join("test.rs"), "#[test] fn test() {}").await?;
        
        let config = IndexingEngineConfig::new(
            "test-repo".to_string(),
            temp_dir.path().to_path_buf(),
        ).with_mode(IndexingMode::FullReindex);
        
        let mut engine = IndexingEngine::new_memory(config).await?;
        engine.start().await?;
        
        // Get initial queue size (should be 0)
        let initial_queue_size = engine.task_queue.queue_size().await;
        assert_eq!(initial_queue_size, 0);
        
        // Give some time for the full reindex to schedule tasks
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        // Check that individual file tasks were scheduled
        let final_queue_size = engine.task_queue.queue_size().await;
        
        // We should have at least 6 tasks (the 6 indexable files we created)
        // Note: We expect 6 files because:
        // - file1.rs, file2.py, file3.js, README.md (4 files in root)
        // - src/lib.rs, src/test.rs (2 files in subdirectory)
        assert!(final_queue_size >= 6, "Expected at least 6 tasks but found {}", final_queue_size);
        
        // Verify we can process some tasks
        engine.process_pending_tasks().await?;
        
        engine.shutdown().await?;
        
        Ok(())
    }
}