//! Core status types and comprehensive reporting structures for janet-ai-retriever
//!
//! This module provides the main data structures for comprehensive system status reporting,
//! including index statistics, indexing progress, health checks, and configuration details.
//! It serves as the central type system for the status API.
//!
//! ## Key Components
//!
//! - **ComprehensiveStatus**: Main status report aggregating all subsystem information
//! - **IndexStatistics**: Statistics about files, chunks, and embeddings in the index
//! - **IndexingStatus**: Real-time indexing operation progress and queue status
//! - **IndexHealth**: System health checks including database connectivity and resources
//! - **IndexingConfiguration**: Current indexing settings and parameters
//! - **EmbeddingModelInfo**: Embedding model details and download status
//!
//! ## Usage
//!

use super::{
    DatabaseInfo, DependencyVersions, FileSystemStatus, IndexConsistencyReport,
    IndexingPerformanceStats, NetworkStatus, SearchPerformanceStats, StaleFilesInfo,
};
use serde::{Deserialize, Serialize};

/// Comprehensive system status aggregating all subsystem reports. See module docs for usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveStatus {
    /// Index statistics (files, chunks, embeddings)
    pub index_statistics: Option<IndexStatistics>,
    /// Current indexing operation status
    pub indexing_status: Option<IndexingStatus>,
    /// Index health check information
    pub index_health: Option<IndexHealth>,
    /// Current indexing configuration
    pub indexing_configuration: Option<IndexingConfiguration>,
    /// Embedding model information
    pub embedding_model_info: Option<EmbeddingModelInfo>,
    /// Database information and statistics
    pub database_info: Option<DatabaseInfo>,
    /// Dependency versions
    pub dependency_versions: Option<DependencyVersions>,
    /// Index consistency validation report
    pub consistency_report: Option<IndexConsistencyReport>,
    /// File system monitoring status
    pub file_system_status: Option<FileSystemStatus>,
    /// Search performance statistics
    pub search_performance: Option<SearchPerformanceStats>,
    /// Indexing performance statistics
    pub indexing_performance: Option<IndexingPerformanceStats>,
    /// Stale files information
    pub stale_files: Option<StaleFilesInfo>,
    /// Network connectivity status
    pub network_status: Option<NetworkStatus>,
    /// List of supported file types
    pub supported_file_types: Option<Vec<String>>,
}

impl ComprehensiveStatus {
    /// Creates a new empty status report. See module docs for usage examples.
    pub fn new() -> Self {
        Self {
            index_statistics: None,
            indexing_status: None,
            index_health: None,
            indexing_configuration: None,
            embedding_model_info: None,
            database_info: None,
            dependency_versions: None,
            consistency_report: None,
            file_system_status: None,
            search_performance: None,
            indexing_performance: None,
            stale_files: None,
            network_status: None,
            supported_file_types: None,
        }
    }

    /// Serializes status report to TOML format. See module docs for details.
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

impl Default for ComprehensiveStatus {
    fn default() -> Self {
        Self::new()
    }
}

/// Index statistics and metadata. See module docs for details.
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

/// Real-time indexing progress and queue status. See module docs for details.
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

/// System health checks and resource status. See module docs for details.
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

/// Health status levels for system diagnostics. See module docs for details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
}

/// Indexing settings and parameters. See module docs for details.
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

/// Embedding model details and status. See module docs for usage examples.
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
    /// Normalization settings
    pub normalized: bool,
}

/// Model download status tracking. See module docs for details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelDownloadStatus {
    NotDownloaded,
    Downloading,
    Downloaded,
    Failed(String),
}

// Removed OnnxRuntimeInfo - not tracked
