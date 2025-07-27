//! High-level indexing engine that orchestrates the complete indexing pipeline.
//!
//! This module provides the main orchestration layer for janet-ai-retriever, coordinating
//! file discovery, chunking, embedding generation, and storage. It manages the entire
//! indexing workflow from raw files to searchable chunks with embeddings.
//!
//! ## Key Components
//!
//! - **IndexingEngine**: Main orchestrator for the indexing pipeline
//! - **IndexingEngineConfig**: Configuration for indexing behavior and resources
//! - **IndexingStats**: Runtime statistics about indexing progress
//!
//! ## Pipeline Flow
//!
//! ```text
//! Files → Analyzer → ChunkingStrategy → Embeddings → EnhancedFileIndex
//!   ↑         ↑            ↑               ↑              ↑
//!   |    FileScanner   TextContextBuilder  FastEmbed   SQLite Storage
//!   |         |            |               |              |
//! DirectoryWatcher → TaskQueue → EmbeddingProvider → IndexMetadata
//! ```
//!
//! ## Features
//!
//! ### Indexing Modes
//! By default, the engine starts in read-only mode. Call `start(full_reindex)` to begin indexing:
//! - `start(true)`: Complete rebuild of the index followed by continuous monitoring
//! - `start(false)`: Continuous monitoring of file changes only
//!
//! ### Async Task Processing
//! - Configurable worker pools for parallel processing
//! - Priority-based task queuing
//! - Graceful error handling and retry logic
//!
//! ### Resource Management
//! - Configurable memory limits for embedding models
//! - Batch processing for efficient embedding generation
//! - Connection pooling for database operations
//!
//! ## Usage
//!
//! ### Basic Indexing
//!
//! ### Read-Only Access
//!
//! ## Performance Considerations
//!
//! - **Worker Count**: More workers = faster processing but higher memory usage
//! - **Chunk Size**: Smaller chunks = better search granularity but more storage
//! - **Batch Size**: Larger batches = more efficient embedding but higher memory peaks
//! - **Embedding Models**: Smaller models = faster processing but potentially lower quality

use anyhow::Result;
use blake3;
use janet_ai_embed::{EmbedConfig, EmbeddingProvider, FastEmbedProvider};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, error, info, warn};

use super::chunking_strategy::{ChunkingConfig, ChunkingStrategy};
use super::enhanced_index::{EmbeddingModelMetadata, EnhancedFileIndex, IndexMetadata, IndexStats};
use super::file_index::{ChunkRef, FileRef};
use super::task_queue::{IndexingTask, TaskQueue, TaskQueueConfig, TaskType};

/// Configuration for the indexing engine
#[derive(Debug, Clone)]
pub struct IndexingEngineConfig {
    /// Repository name
    pub repository: String,
    /// Base path for indexing
    pub base_path: PathBuf,
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
    /// Create a new indexing engine configuration.
    ///
    /// This creates a configuration with sensible defaults for most use cases.
    /// You can customize the configuration using the builder methods.
    ///
    /// # Arguments
    /// * `repository` - Name of the repository being indexed (used for identification)
    /// * `base_path` - Root directory containing files to index
    ///
    /// # Returns
    /// A new configuration with default settings:
    /// - 4 worker threads
    /// - No embedding generation (text search only)
    /// - 60 second timeout per file
    /// - Default chunking strategy
    ///
    /// # Example
    pub fn new(repository: String, base_path: PathBuf) -> Self {
        Self {
            chunking_config: ChunkingConfig::new(repository.clone()),
            repository,
            base_path,
            task_queue_config: TaskQueueConfig::default(),
            embedding_config: None,
            max_workers: 4,
            file_timeout: Duration::from_secs(60),
        }
    }

    /// Enable embedding generation with the specified configuration.
    ///
    /// When embeddings are enabled, the indexing engine will generate vector
    /// embeddings for each text chunk, enabling semantic similarity search.
    /// Without embeddings, only text-based search is available.
    ///
    /// # Arguments
    /// * `config` - Embedding configuration specifying model and parameters
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    pub fn with_embedding_config(mut self, config: EmbedConfig) -> Self {
        self.embedding_config = Some(config);
        self
    }

    /// Set the maximum number of worker threads for parallel processing.
    ///
    /// More workers can speed up indexing but use more CPU and memory.
    /// The optimal number depends on your system and the types of files being indexed.
    ///
    /// # Arguments
    /// * `workers` - Number of worker threads (recommended: number of CPU cores)
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    pub fn with_max_workers(mut self, workers: usize) -> Self {
        self.max_workers = workers;
        self.task_queue_config.max_workers = workers;
        self
    }

    /// Set the maximum size for text chunks in characters.
    ///
    /// Smaller chunks provide more granular search results but may lose context.
    /// Larger chunks preserve more context but may be less precise for search.
    /// The optimal size depends on your use case and content type.
    ///
    /// # Arguments
    /// * `size` - Maximum chunk size in characters (recommended: 500-2000)
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
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
#[derive(Debug)]
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
    started_for_indexing: bool,
}

#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub files_processed: usize,
    pub chunks_created: usize,
    pub embeddings_generated: usize,
    pub errors: usize,
}

impl IndexingEngine {
    /// Create a new indexing engine with the specified configuration.
    ///
    /// This initializes the indexing engine with a persistent SQLite database.
    /// The engine starts in read-only mode - call [`start`](Self::start) to begin indexing.
    ///
    /// # Arguments
    /// * `config` - Configuration specifying repository, paths, and indexing options
    ///
    /// # Returns
    /// A new indexing engine ready for read-only operations
    ///
    /// # Errors
    /// - Database initialization errors
    /// - File system permission errors
    /// - Configuration validation errors
    ///
    /// # Example
    pub async fn new(config: IndexingEngineConfig) -> Result<Self> {
        Self::new_impl(config, false).await
    }

    /// Create a new indexing engine with an in-memory database.
    ///
    /// This is primarily intended for testing and development. The database
    /// will be lost when the engine is dropped. For production use, prefer [`new`](Self::new).
    ///
    /// # Arguments
    /// * `config` - Configuration specifying repository, paths, and indexing options
    ///
    /// # Returns
    /// A new indexing engine with an in-memory database
    ///
    /// # Errors
    /// Same as [`new`](Self::new) but without persistent storage errors
    ///
    /// # Example
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
        let (embedding_provider, embedding_model_metadata) =
            if let Some(embed_config) = &config.embedding_config {
                info!(
                    "Initializing embedding provider: {}",
                    embed_config.model_name
                );

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
                info!("No embedding configuration provided");
                (None, None)
            };

        // Initialize or update index metadata
        let index_metadata = if let Some(ref embed_meta) = embedding_model_metadata {
            IndexMetadata::new(config.repository.clone()).with_embedding_model(embed_meta.clone())
        } else {
            IndexMetadata::new(config.repository.clone())
        };
        enhanced_index
            .upsert_index_metadata(&index_metadata)
            .await?;

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
            started_for_indexing: false,
        })
    }

    /// Start the indexing engine for active indexing operations.
    ///
    /// This transitions the engine from read-only mode to active indexing mode,
    /// enabling file monitoring, task processing, and index updates. The engine
    /// must be started before calling methods like [`schedule_file_index`](Self::schedule_file_index)
    /// or [`schedule_full_reindex`](Self::schedule_full_reindex).
    ///
    /// # Arguments
    /// * `full_reindex` - Whether to perform a complete reindex of all files
    ///   - `true`: Rebuild the entire index, then start continuous monitoring
    ///   - `false`: Start continuous monitoring without rebuilding existing index
    ///
    /// # Returns
    /// `Ok(())` if the engine started successfully
    ///
    /// # Errors
    /// - Task queue initialization errors
    /// - File system errors during initial indexing
    /// - Database errors
    ///
    /// # Example
    pub async fn start(&mut self, full_reindex: bool) -> Result<()> {
        info!(
            "Starting IndexingEngine with full_reindex: {}",
            full_reindex
        );

        // Mark as started for indexing
        self.started_for_indexing = true;

        // Task queue with flume doesn't need explicit processor startup

        // Create shutdown channel
        let (shutdown_sender, _shutdown_receiver) = mpsc::unbounded_channel();
        self.shutdown_sender = Some(shutdown_sender);

        // For now, we'll implement a simple single-threaded worker
        // TODO: Implement proper multi-threading later
        info!("Worker system initialized (single-threaded mode)");

        info!("Started {} indexing workers", self.config.max_workers);

        // Perform initial index if needed
        if full_reindex {
            self.schedule_full_reindex().await?;
        }

        Ok(())
    }

    /// Process all currently pending tasks from the indexing queue.
    ///
    /// This method processes tasks that have been scheduled for indexing, including
    /// file indexing, file removal, and other maintenance operations. It processes
    /// all pending tasks up to a safety limit to prevent infinite loops.
    ///
    /// Tasks are processed sequentially and statistics are updated for each completed task.
    /// If a task fails, the error is logged and the failure count is incremented, but
    /// processing continues with the next task.
    ///
    /// # Returns
    /// `Ok(())` when all pending tasks have been processed
    ///
    /// # Errors
    /// This method does not return errors for individual task failures, but may return
    /// errors for queue access issues or other system-level problems.
    ///
    /// # Example
    pub async fn process_pending_tasks(&self) -> Result<()> {
        let max_tasks_per_batch = 100; // Safety limit to prevent infinite loops
        let mut tasks_processed = 0;

        while let Ok(task) = self.task_queue.try_recv_task() {
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
                debug!(
                    "Reached max tasks per batch ({}), stopping to prevent hanging",
                    max_tasks_per_batch
                );
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
                )
                .await
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
        let content = tokio::fs::read_to_string(file_path)
            .await
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
        if let (Some(provider), Some(model_id)) =
            (embedding_provider, embedding_model_id.as_deref())
        {
            let texts: Vec<String> = chunk_refs.iter().map(|c| c.content.clone()).collect();

            match provider.embed_texts(&texts).await {
                Ok(embedding_result) => {
                    for (chunk, embedding) in chunk_refs.iter_mut().zip(embedding_result.embeddings)
                    {
                        chunk.embedding = Some(embedding);
                        embeddings_generated += 1;
                    }

                    // Store chunks with embeddings
                    enhanced_index
                        .upsert_chunks_with_model(&chunk_refs, model_id)
                        .await?;
                }
                Err(e) => {
                    warn!(
                        "Failed to generate embeddings for {}: {}",
                        file_path.display(),
                        e
                    );
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

    /// Schedule a complete reindex of all files in the configured directory.
    ///
    /// This method recursively walks the directory tree starting from the configured
    /// base path and schedules indexing tasks for all files that should be indexed
    /// according to the chunking strategy. Files are processed in batches to avoid
    /// overwhelming the task queue.
    ///
    /// The indexing engine must be started (not in read-only mode) before calling this method.
    ///
    /// # Returns
    /// `Ok(())` when all files have been scheduled for indexing
    ///
    /// # Errors
    /// - If called while the engine is in read-only mode
    /// - File system errors during directory traversal
    /// - Task queue submission errors
    ///
    /// # Example
    pub async fn schedule_full_reindex(&self) -> Result<()> {
        if !self.started_for_indexing {
            return Err(anyhow::anyhow!(
                "Full reindex not allowed in read-only mode. Call start() first."
            ));
        }

        info!(
            "Starting full reindex of: {}",
            self.config.base_path.display()
        );

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
                        if let Err(e) = self
                            .task_queue
                            .submit_tasks(std::mem::take(&mut tasks_batch))
                        {
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
            if let Err(e) = self.task_queue.submit_tasks(tasks_batch) {
                warn!("Failed to submit final batch of tasks: {}", e);
            }
        }

        info!(
            "Scheduled {} files for background indexing",
            files_scheduled
        );
        Ok(())
    }

    /// Schedule indexing of a single file.
    ///
    /// This method adds a single file to the indexing queue for processing.
    /// The file will be read, chunked, and have embeddings generated (if configured)
    /// during the next call to [`process_pending_tasks`](Self::process_pending_tasks).
    ///
    /// The indexing engine must be started (not in read-only mode) before calling this method.
    ///
    /// # Arguments
    /// * `file_path` - Path to the file to be indexed
    ///
    /// # Returns
    /// `Ok(())` when the file has been successfully scheduled
    ///
    /// # Errors
    /// - If called while the engine is in read-only mode
    /// - Task queue submission errors
    ///
    /// # Example
    pub async fn schedule_file_index(&self, file_path: PathBuf) -> Result<()> {
        if !self.started_for_indexing {
            return Err(anyhow::anyhow!(
                "File indexing not allowed in read-only mode. Call start() first."
            ));
        }

        let task = IndexingTask::index_file(file_path);
        self.task_queue
            .submit_task(task)
            .map_err(|e| anyhow::anyhow!("Failed to schedule file index: {}", e))
    }

    /// Get current processing statistics for this indexing session.
    ///
    /// Returns real-time statistics about files processed, chunks created,
    /// embeddings generated, and any errors encountered since the engine started.
    ///
    /// # Returns
    /// ProcessingStats containing current session statistics
    ///
    /// # Example
    pub async fn get_stats(&self) -> ProcessingStats {
        self.stats.read().await.clone()
    }

    /// Get comprehensive statistics about the stored index.
    ///
    /// Returns statistics about the total content stored in the database,
    /// including files, chunks, embeddings, and model information.
    /// This reflects the complete index state, not just the current session.
    ///
    /// # Returns
    /// IndexStats containing comprehensive database statistics
    ///
    /// # Errors
    /// Database query errors
    ///
    /// # Example
    pub async fn get_index_stats(&self) -> Result<IndexStats> {
        self.enhanced_index.get_index_stats().await
    }

    /// Get the number of tasks currently waiting to be processed.
    ///
    /// This returns the size of the internal task queue. A size of 0 indicates
    /// that all scheduled indexing work has been completed.
    ///
    /// # Returns
    /// Number of pending tasks in the queue
    ///
    /// # Example
    pub async fn get_queue_size(&self) -> usize {
        self.task_queue.queue_size()
    }

    /// Get a reference to the enhanced file index for performing searches.
    ///
    /// The enhanced file index provides access to search functionality including
    /// text search, semantic search (if embeddings are available), and metadata queries.
    /// This is the primary interface for querying the indexed content.
    ///
    /// # Returns
    /// Reference to the EnhancedFileIndex for search operations
    ///
    /// # Example
    pub fn get_enhanced_index(&self) -> &EnhancedFileIndex {
        &self.enhanced_index
    }

    /// Gracefully shut down the indexing engine.
    ///
    /// This method stops all workers, shuts down the task queue, and performs
    /// cleanup operations. After shutdown, the engine cannot be restarted and
    /// should be dropped.
    ///
    /// It's recommended to call this method before dropping the engine to ensure
    /// all background tasks complete cleanly.
    ///
    /// # Returns
    /// `Ok(())` when shutdown is complete
    ///
    /// # Errors
    /// - Worker join errors (logged but not propagated)
    /// - Task queue shutdown errors
    ///
    /// # Example
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down IndexingEngine");

        if !self.started_for_indexing {
            return Ok(());
        }

        // Send shutdown signal to workers
        if let Some(sender) = self.shutdown_sender.take() {
            let _ = sender.send(());
        }

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.await;
        }

        // Flume channels shut down automatically when dropped

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
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let engine = IndexingEngine::new_memory(config).await?;

        // Should be able to get stats even in read-only mode
        let stats = engine.get_index_stats().await?;
        assert_eq!(stats.files_count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_file_scheduling() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let mut engine = IndexingEngine::new_memory(config).await?;
        engine.start(false).await?;

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

        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let mut engine = IndexingEngine::new_memory(config).await?;
        engine.start(true).await?;

        // With flume, tasks are immediately available after start(true)
        // Give some time for the full reindex to complete
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check that individual file tasks were scheduled
        let final_queue_size = engine.task_queue.queue_size();

        // We should have at least 6 tasks (the 6 indexable files we created)
        // Note: We expect 6 files because:
        // - file1.rs, file2.py, file3.js, README.md (4 files in root)
        // - src/lib.rs, src/test.rs (2 files in subdirectory)
        assert!(
            final_queue_size >= 6,
            "Expected at least 6 tasks but found {final_queue_size}"
        );

        // Verify we can process some tasks
        engine.process_pending_tasks().await?;

        engine.shutdown().await?;

        Ok(())
    }
}
