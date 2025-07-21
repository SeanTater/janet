use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::retrieval::{
    enhanced_index::{EmbeddingModelMetadata, EnhancedFileIndex},
    indexing_engine::{IndexingEngine, IndexingEngineConfig},
};

/// Comprehensive index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    /// Total files indexed
    pub total_files: usize,
    /// Total chunks created
    pub total_chunks: usize,
    /// Total embeddings generated
    pub total_embeddings: usize,
    /// Index database size in bytes
    pub database_size_bytes: Option<u64>,
    /// Last indexing operation timestamp (Unix timestamp)
    pub last_indexing_timestamp: Option<i64>,
    /// Index schema version
    pub schema_version: String,
    /// Number of embedding models registered
    pub models_count: usize,
}

/// Current indexing operation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingStatus {
    /// Is indexing currently running?
    pub is_running: bool,
    /// Queue size (pending files to process)
    pub queue_size: usize,
    /// Current file being processed (if any)
    pub current_file: Option<String>,
    /// Progress percentage (0-100)
    pub progress_percentage: Option<f32>,
    /// Estimated time remaining in seconds
    pub estimated_time_remaining_seconds: Option<u64>,
    /// Error count during current/last run
    pub error_count: usize,
    /// Total files processed in current session
    pub files_processed: usize,
    /// Chunks created in current session
    pub chunks_created: usize,
    /// Embeddings generated in current session
    pub embeddings_generated: usize,
}

/// Index health check information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHealth {
    /// Database connectivity status
    pub database_connected: bool,
    /// Database connectivity error message (if any)
    pub database_error: Option<String>,
    /// File permissions for index directory
    pub index_directory_writable: bool,
    /// Disk space available for index growth in bytes
    pub available_disk_space_bytes: Option<u64>,
    /// Memory usage estimates in bytes
    pub estimated_memory_usage_bytes: Option<u64>,
    /// Database integrity check results
    pub database_integrity_ok: bool,
    /// Overall health status
    pub overall_status: HealthStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
}

/// Current indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfiguration {
    /// Chunk size settings
    pub max_chunk_size: usize,
    /// Chunk overlap settings
    pub chunk_overlap: usize,
    /// File type patterns included
    pub included_file_patterns: Vec<String>,
    /// File type patterns excluded
    pub excluded_file_patterns: Vec<String>,
    /// Maximum file size limits in bytes
    pub max_file_size_bytes: Option<u64>,
    /// Indexing mode
    pub indexing_mode: String,
    /// Worker thread count
    pub worker_thread_count: usize,
    /// Repository name
    pub repository: String,
    /// Base path being indexed
    pub base_path: String,
}

/// Embedding model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    /// Model name and version
    pub model_name: String,
    /// Model provider
    pub provider: String,
    /// Model dimensions
    pub dimensions: usize,
    /// Model download status
    pub download_status: ModelDownloadStatus,
    /// Model file size in bytes
    pub model_file_size_bytes: Option<u64>,
    /// Model file location
    pub model_file_location: Option<String>,
    /// Supported languages/domains
    pub supported_languages: Vec<String>,
    /// Normalization settings
    pub normalized: bool,
    /// ONNX runtime details
    pub onnx_runtime_info: Option<OnnxRuntimeInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelDownloadStatus {
    NotDownloaded,
    Downloading,
    Downloaded,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxRuntimeInfo {
    /// Whether GPU acceleration is available
    pub gpu_available: bool,
    /// GPU device name (if available)
    pub gpu_device: Option<String>,
    /// ONNX runtime version
    pub runtime_version: Option<String>,
}

/// Main status API implementation
pub struct StatusApi;

impl StatusApi {
    /// Get comprehensive index statistics
    pub async fn get_index_statistics(
        enhanced_index: &EnhancedFileIndex,
    ) -> Result<IndexStatistics> {
        let index_stats = enhanced_index.get_index_stats().await?;

        // Get database file size if possible
        let database_size_bytes = Self::get_database_size(enhanced_index).await.ok();

        // Get last indexing timestamp from metadata
        let last_indexing_timestamp = Self::get_last_indexing_timestamp(enhanced_index).await.ok();

        Ok(IndexStatistics {
            total_files: index_stats.files_count,
            total_chunks: index_stats.chunks_count,
            total_embeddings: index_stats.embeddings_count,
            database_size_bytes,
            last_indexing_timestamp,
            schema_version: "1.0.0".to_string(), // TODO: Make this dynamic
            models_count: index_stats.models_count,
        })
    }

    /// Get current indexing operation status
    pub async fn get_indexing_status(engine: &IndexingEngine) -> Result<IndexingStatus> {
        let processing_stats = engine.get_stats().await;
        let queue_size = engine.get_queue_size().await;

        // For now, we don't have a way to determine if indexing is currently running
        // This would require additional state tracking in the IndexingEngine
        let is_running = queue_size > 0;

        Ok(IndexingStatus {
            is_running,
            queue_size,
            current_file: None,        // TODO: Track current file being processed
            progress_percentage: None, // TODO: Calculate based on queue progress
            estimated_time_remaining_seconds: None, // TODO: Estimate based on processing rate
            error_count: processing_stats.errors,
            files_processed: processing_stats.files_processed,
            chunks_created: processing_stats.chunks_created,
            embeddings_generated: processing_stats.embeddings_generated,
        })
    }

    /// Get index health check information
    pub async fn get_index_health(enhanced_index: &EnhancedFileIndex) -> Result<IndexHealth> {
        let mut health = IndexHealth {
            database_connected: false,
            database_error: None,
            index_directory_writable: false,
            available_disk_space_bytes: None,
            estimated_memory_usage_bytes: None,
            database_integrity_ok: false,
            overall_status: HealthStatus::Critical,
        };

        // Test database connectivity
        match enhanced_index.get_index_stats().await {
            Ok(_) => {
                health.database_connected = true;
                health.database_integrity_ok = true;
            }
            Err(e) => {
                health.database_error = Some(e.to_string());
            }
        }

        // TODO: Implement additional health checks:
        // - Check directory permissions
        // - Check available disk space
        // - Estimate memory usage
        // - Run database integrity checks

        // Determine overall status
        health.overall_status = if health.database_connected && health.database_integrity_ok {
            HealthStatus::Healthy
        } else if health.database_connected {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };

        Ok(health)
    }

    /// Get current indexing configuration
    pub async fn get_indexing_config(
        config: &IndexingEngineConfig,
    ) -> Result<IndexingConfiguration> {
        Ok(IndexingConfiguration {
            max_chunk_size: config.chunking_config.max_chunk_size,
            chunk_overlap: 0, // TODO: ChunkingConfig doesn't currently have overlap
            included_file_patterns: Vec::new(), // TODO: Add file pattern support to ChunkingStrategy
            excluded_file_patterns: Vec::new(), // TODO: Add file pattern support to ChunkingStrategy
            max_file_size_bytes: None,          // TODO: Add to chunking config
            indexing_mode: format!("{}", config.mode),
            worker_thread_count: config.max_workers,
            repository: config.repository.clone(),
            base_path: config.base_path.to_string_lossy().to_string(),
        })
    }

    /// Get embedding model information
    pub async fn get_embedding_model_info(
        metadata: Option<&EmbeddingModelMetadata>,
    ) -> Result<Option<EmbeddingModelInfo>> {
        let Some(metadata) = metadata else {
            return Ok(None);
        };

        Ok(Some(EmbeddingModelInfo {
            model_name: metadata.model_name.clone(),
            provider: metadata.provider.clone(),
            dimensions: metadata.dimension,
            download_status: ModelDownloadStatus::Downloaded, // TODO: Implement proper status check
            model_file_size_bytes: None,                      // TODO: Get from filesystem
            model_file_location: None,                        // TODO: Get from FastEmbed provider
            supported_languages: vec!["multilingual".to_string()], // TODO: Make model-specific
            normalized: metadata.normalized,
            onnx_runtime_info: Some(OnnxRuntimeInfo {
                gpu_available: false, // TODO: Check actual GPU availability
                gpu_device: None,
                runtime_version: None,
            }),
        }))
    }

    /// Get list of supported file types
    pub async fn get_supported_file_types(_config: &IndexingEngineConfig) -> Result<Vec<String>> {
        // TODO: Get this from ChunkingStrategy.should_index_file logic
        Ok(vec![
            "rs".to_string(),
            "py".to_string(),
            "js".to_string(),
            "ts".to_string(),
            "jsx".to_string(),
            "tsx".to_string(),
            "java".to_string(),
            "cpp".to_string(),
            "c".to_string(),
            "h".to_string(),
            "hpp".to_string(),
            "go".to_string(),
            "rb".to_string(),
            "php".to_string(),
            "cs".to_string(),
            "swift".to_string(),
            "kt".to_string(),
            "scala".to_string(),
            "md".to_string(),
            "txt".to_string(),
            "json".to_string(),
            "yaml".to_string(),
            "yml".to_string(),
            "toml".to_string(),
            "xml".to_string(),
            "html".to_string(),
            "css".to_string(),
            "scss".to_string(),
            "less".to_string(),
            "sql".to_string(),
        ])
    }

    // Helper methods
    async fn get_database_size(_enhanced_index: &EnhancedFileIndex) -> Result<u64> {
        // TODO: Implement database size calculation
        // This would involve getting the SQLite database file path and checking its size
        Ok(0)
    }

    async fn get_last_indexing_timestamp(_enhanced_index: &EnhancedFileIndex) -> Result<i64> {
        // TODO: Implement by querying index metadata
        // For now, return current timestamp
        Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{
        indexing_engine::{IndexingEngine, IndexingEngineConfig},
        indexing_mode::IndexingMode,
    };
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_get_index_statistics() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf())
                .with_mode(IndexingMode::ReadOnly);

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let stats = StatusApi::get_index_statistics(enhanced_index).await?;

        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_embeddings, 0);
        assert_eq!(stats.models_count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_indexing_status() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf())
                .with_mode(IndexingMode::ReadOnly);

        let engine = IndexingEngine::new_memory(config).await?;

        let status = StatusApi::get_indexing_status(&engine).await?;

        assert!(!status.is_running);
        assert_eq!(status.queue_size, 0);
        assert_eq!(status.error_count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_index_health() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf())
                .with_mode(IndexingMode::ReadOnly);

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let health = StatusApi::get_index_health(enhanced_index).await?;

        assert!(health.database_connected);
        assert!(health.database_integrity_ok);
        assert!(matches!(health.overall_status, HealthStatus::Healthy));

        Ok(())
    }
}
