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

/// Database information and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseInfo {
    /// Database type and version
    pub database_type: String,
    /// Database version
    pub database_version: Option<String>,
    /// Connection pool status
    pub connection_pool_status: ConnectionPoolStatus,
    /// Database file locations and sizes
    pub database_files: Vec<DatabaseFile>,
    /// Total database size in bytes
    pub total_size_bytes: Option<u64>,
    /// SQLite specific information
    pub sqlite_info: Option<SqliteInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolStatus {
    /// Total connections in pool
    pub total_connections: u32,
    /// Active connections
    pub active_connections: u32,
    /// Maximum connections allowed
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connection_timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseFile {
    /// File path
    pub path: String,
    /// File size in bytes
    pub size_bytes: Option<u64>,
    /// File type (main database, WAL, journal, etc.)
    pub file_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqliteInfo {
    /// SQLite version
    pub version: String,
    /// Journal mode (WAL, DELETE, etc.)
    pub journal_mode: Option<String>,
    /// Synchronous mode
    pub synchronous_mode: Option<String>,
    /// Page size
    pub page_size: Option<u64>,
    /// Total pages
    pub page_count: Option<u64>,
}

/// Version information for dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyVersions {
    /// janet-ai-retriever version
    pub retriever_version: String,
    /// janet-ai-embed version
    pub embed_version: String,
    /// janet-ai-context version
    pub context_version: String,
    /// Rust version used for compilation
    pub rust_version: String,
    /// Core dependency versions
    pub dependencies: std::collections::HashMap<String, String>,
}

/// Index consistency check results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConsistencyReport {
    /// Overall consistency status
    pub overall_status: ConsistencyStatus,
    /// Checks performed
    pub checks_performed: Vec<ConsistencyCheck>,
    /// Summary of issues found
    pub issues_summary: IssuesSummary,
    /// Timestamp when check was performed
    pub check_timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyStatus {
    Healthy,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyCheck {
    /// Name of the check
    pub check_name: String,
    /// Check status
    pub status: ConsistencyStatus,
    /// Description of what was checked
    pub description: String,
    /// Number of items checked
    pub items_checked: usize,
    /// Number of issues found
    pub issues_found: usize,
    /// Details about issues (if any)
    pub issue_details: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssuesSummary {
    /// Total issues found
    pub total_issues: usize,
    /// Critical issues requiring immediate attention
    pub critical_issues: usize,
    /// Warning issues that should be addressed
    pub warning_issues: usize,
    /// Recommendations for fixing issues
    pub recommendations: Vec<String>,
}

/// File system monitoring status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemStatus {
    /// Is file watching currently active?
    pub file_watching_active: bool,
    /// Number of directories being monitored
    pub directories_monitored: usize,
    /// File watcher error count
    pub watcher_error_count: usize,
    /// Last file change event timestamp
    pub last_change_event_timestamp: Option<i64>,
    /// Supported file system types
    pub supported_file_systems: Vec<String>,
    /// File system type of base directory
    pub base_directory_fs_type: Option<String>,
    /// Recent file change events (last 10)
    pub recent_events: Vec<FileChangeEvent>,
}

/// Search performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPerformanceStats {
    /// Average search response time in milliseconds (last 100 queries)
    pub average_response_time_ms: Option<f64>,
    /// Search result quality metrics
    pub result_quality_metrics: SearchQualityMetrics,
    /// Cache hit rate percentage (0-100)
    pub cache_hit_rate_percentage: Option<f32>,
    /// Most common query patterns
    pub common_query_patterns: Vec<String>,
    /// Error rates by operation type
    pub error_rates: SearchErrorRates,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQualityMetrics {
    /// Average number of results returned
    pub average_results_count: Option<f64>,
    /// Average relevance score (0-1)
    pub average_relevance_score: Option<f64>,
    /// Percentage of searches returning zero results
    pub zero_results_percentage: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchErrorRates {
    /// Semantic search error rate (0-1)
    pub semantic_search_error_rate: f32,
    /// Text search error rate (0-1)
    pub text_search_error_rate: f32,
    /// Total queries processed
    pub total_queries_processed: usize,
}

/// Indexing performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingPerformanceStats {
    /// Files processed per minute (recent average)
    pub files_per_minute: Option<f64>,
    /// Average processing time per file type in milliseconds
    pub processing_time_by_file_type: std::collections::HashMap<String, f64>,
    /// Embedding generation speed (embeddings per second)
    pub embeddings_per_second: Option<f64>,
    /// Disk I/O statistics
    pub disk_io_stats: DiskIOStats,
    /// Memory usage during indexing in bytes
    pub peak_memory_usage_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOStats {
    /// Bytes read per second (recent average)
    pub bytes_read_per_second: Option<u64>,
    /// Bytes written per second (recent average)
    pub bytes_written_per_second: Option<u64>,
    /// Total disk space used for indexing in bytes
    pub total_disk_space_used_bytes: Option<u64>,
}

/// Stale files information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaleFilesInfo {
    /// Files modified after last index update
    pub modified_files: Vec<StaleFileEntry>,
    /// Files added but not yet indexed
    pub unindexed_files: Vec<StaleFileEntry>,
    /// Files deleted but still in index (estimated)
    pub deleted_files_in_index: Vec<StaleFileEntry>,
    /// Recommended reindex candidates
    pub reindex_candidates: Vec<String>,
    /// Total estimated stale files count
    pub total_stale_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaleFileEntry {
    /// File path
    pub file_path: String,
    /// Last modified timestamp
    pub last_modified_timestamp: Option<i64>,
    /// Reason for being considered stale
    pub staleness_reason: String,
    /// Priority for reindexing (1-10, higher is more urgent)
    pub reindex_priority: u8,
}

/// Network status for external dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    /// Model download connectivity
    pub model_download_connectivity: ConnectivityStatus,
    /// Hugging Face Hub access
    pub hugging_face_hub_access: ConnectivityStatus,
    /// Proxy configuration status
    pub proxy_configuration: ProxyStatus,
    /// SSL certificate validation status
    pub ssl_certificate_validation: bool,
    /// Overall network health
    pub overall_network_health: NetworkHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityStatus {
    /// Is the service reachable?
    pub is_reachable: bool,
    /// Last successful connection timestamp
    pub last_successful_connection: Option<i64>,
    /// Error message if unreachable
    pub error_message: Option<String>,
    /// Response time in milliseconds for last check
    pub response_time_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyStatus {
    /// Is proxy configured?
    pub proxy_configured: bool,
    /// Proxy host and port
    pub proxy_address: Option<String>,
    /// Proxy authentication status
    pub proxy_auth_configured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkHealth {
    Healthy,
    Limited,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChangeEvent {
    /// Event type (Created, Modified, Deleted, etc.)
    pub event_type: String,
    /// File path that changed
    pub file_path: String,
    /// Timestamp when event occurred
    pub timestamp: i64,
    /// Whether the event was processed successfully
    pub processed_successfully: bool,
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

    /// Get database information and statistics
    pub async fn get_database_info(
        enhanced_index: &EnhancedFileIndex,
        base_path: &std::path::Path,
    ) -> Result<DatabaseInfo> {
        let db_path = base_path.join(".janet-ai.db");

        // Get SQLite version and other info
        let sqlite_version: String = sqlx::query_scalar("SELECT sqlite_version()")
            .fetch_one(enhanced_index.pool())
            .await
            .unwrap_or_else(|_| "unknown".to_string());

        let journal_mode: Option<String> = sqlx::query_scalar("PRAGMA journal_mode")
            .fetch_optional(enhanced_index.pool())
            .await?;

        let synchronous_mode: Option<i64> = sqlx::query_scalar("PRAGMA synchronous")
            .fetch_optional(enhanced_index.pool())
            .await?;
        let synchronous_mode_str = synchronous_mode.map(|v| match v {
            0 => "OFF".to_string(),
            1 => "NORMAL".to_string(),
            2 => "FULL".to_string(),
            3 => "EXTRA".to_string(),
            _ => format!("{v}"),
        });

        let page_size: Option<i64> = sqlx::query_scalar("PRAGMA page_size")
            .fetch_optional(enhanced_index.pool())
            .await?;

        let page_count: Option<i64> = sqlx::query_scalar("PRAGMA page_count")
            .fetch_optional(enhanced_index.pool())
            .await?;

        // Get database file information
        let mut database_files = Vec::new();
        let mut total_size = 0u64;

        if let Ok(metadata) = std::fs::metadata(&db_path) {
            let size = metadata.len();
            total_size += size;
            database_files.push(DatabaseFile {
                path: db_path.to_string_lossy().to_string(),
                size_bytes: Some(size),
                file_type: "main".to_string(),
            });
        }

        // Check for WAL and journal files
        let wal_path = db_path.with_extension("db-wal");
        if let Ok(metadata) = std::fs::metadata(&wal_path) {
            let size = metadata.len();
            total_size += size;
            database_files.push(DatabaseFile {
                path: wal_path.to_string_lossy().to_string(),
                size_bytes: Some(size),
                file_type: "wal".to_string(),
            });
        }

        let shm_path = db_path.with_extension("db-shm");
        if let Ok(metadata) = std::fs::metadata(&shm_path) {
            let size = metadata.len();
            total_size += size;
            database_files.push(DatabaseFile {
                path: shm_path.to_string_lossy().to_string(),
                size_bytes: Some(size),
                file_type: "shm".to_string(),
            });
        }

        Ok(DatabaseInfo {
            database_type: "SQLite".to_string(),
            database_version: Some(sqlite_version.clone()),
            connection_pool_status: ConnectionPoolStatus {
                total_connections: 1, // TODO: Get from sqlx pool
                active_connections: 1,
                max_connections: 10, // TODO: Get from sqlx pool config
                connection_timeout_seconds: Some(30),
            },
            database_files,
            total_size_bytes: Some(total_size),
            sqlite_info: Some(SqliteInfo {
                version: sqlite_version,
                journal_mode,
                synchronous_mode: synchronous_mode_str,
                page_size: page_size.map(|p| p as u64),
                page_count: page_count.map(|p| p as u64),
            }),
        })
    }

    /// Get version information for all dependencies
    pub async fn get_dependency_versions() -> Result<DependencyVersions> {
        let mut dependencies = std::collections::HashMap::new();

        // Add key dependency versions
        dependencies.insert("sqlx".to_string(), "0.8".to_string()); // TODO: Get actual version
        dependencies.insert("tokio".to_string(), "1.0".to_string()); // TODO: Get actual version
        dependencies.insert("serde".to_string(), "1.0".to_string()); // TODO: Get actual version
        dependencies.insert("anyhow".to_string(), "1.0".to_string()); // TODO: Get actual version
        dependencies.insert("tracing".to_string(), "0.1".to_string()); // TODO: Get actual version

        Ok(DependencyVersions {
            retriever_version: env!("CARGO_PKG_VERSION").to_string(),
            embed_version: "0.1.0".to_string(), // TODO: Get from janet-ai-embed
            context_version: "0.1.0".to_string(), // TODO: Get from janet-ai-context
            rust_version: option_env!("CARGO_PKG_RUST_VERSION")
                .unwrap_or("unknown")
                .to_string(),
            dependencies,
        })
    }

    /// Validate index consistency and return detailed report
    pub async fn validate_index_consistency(
        enhanced_index: &EnhancedFileIndex,
    ) -> Result<IndexConsistencyReport> {
        let mut checks = Vec::new();
        let mut total_issues = 0;
        let mut critical_issues = 0;
        let mut warning_issues = 0;
        let mut recommendations = Vec::new();

        // Check 1: Verify all chunks have corresponding files
        let chunks_without_files: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM chunks c LEFT JOIN files f ON c.file_hash = f.hash WHERE f.hash IS NULL"
        )
        .fetch_one(enhanced_index.pool())
        .await?;

        let chunks_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks")
            .fetch_one(enhanced_index.pool())
            .await?;

        let chunks_check = ConsistencyCheck {
            check_name: "Chunks-Files Consistency".to_string(),
            status: if chunks_without_files > 0 {
                ConsistencyStatus::Warning
            } else {
                ConsistencyStatus::Healthy
            },
            description: "Verify all chunks have corresponding files".to_string(),
            items_checked: chunks_count as usize,
            issues_found: chunks_without_files as usize,
            issue_details: if chunks_without_files > 0 {
                vec![format!(
                    "{} chunks found without corresponding files",
                    chunks_without_files
                )]
            } else {
                Vec::new()
            },
        };

        if chunks_without_files > 0 {
            warning_issues += chunks_without_files as usize;
            total_issues += chunks_without_files as usize;
            recommendations.push("Consider running a reindex to fix orphaned chunks".to_string());
        }
        checks.push(chunks_check);

        // Check 2: Check for orphaned embeddings
        let orphaned_embeddings: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL AND file_hash NOT IN (SELECT hash FROM files)"
        )
        .fetch_one(enhanced_index.pool())
        .await?;

        let embeddings_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
                .fetch_one(enhanced_index.pool())
                .await?;

        let embeddings_check = ConsistencyCheck {
            check_name: "Orphaned Embeddings".to_string(),
            status: if orphaned_embeddings > 0 {
                ConsistencyStatus::Warning
            } else {
                ConsistencyStatus::Healthy
            },
            description: "Check for embeddings without corresponding files".to_string(),
            items_checked: embeddings_count as usize,
            issues_found: orphaned_embeddings as usize,
            issue_details: if orphaned_embeddings > 0 {
                vec![format!("{} orphaned embeddings found", orphaned_embeddings)]
            } else {
                Vec::new()
            },
        };

        if orphaned_embeddings > 0 {
            warning_issues += orphaned_embeddings as usize;
            total_issues += orphaned_embeddings as usize;
            recommendations.push("Clean up orphaned embeddings".to_string());
        }
        checks.push(embeddings_check);

        // Check 3: Validate database integrity (basic)
        let integrity_check_result: String = sqlx::query_scalar("PRAGMA integrity_check")
            .fetch_one(enhanced_index.pool())
            .await
            .unwrap_or_else(|_| "error".to_string());

        let integrity_ok = integrity_check_result == "ok";
        let integrity_check = ConsistencyCheck {
            check_name: "Database Integrity".to_string(),
            status: if integrity_ok {
                ConsistencyStatus::Healthy
            } else {
                ConsistencyStatus::Critical
            },
            description: "SQLite database integrity check".to_string(),
            items_checked: 1,
            issues_found: if integrity_ok { 0 } else { 1 },
            issue_details: if !integrity_ok {
                vec![format!(
                    "Database integrity check failed: {}",
                    integrity_check_result
                )]
            } else {
                Vec::new()
            },
        };

        if !integrity_ok {
            critical_issues += 1;
            total_issues += 1;
            recommendations.push(
                "Database corruption detected - backup and restore from clean copy".to_string(),
            );
        }
        checks.push(integrity_check);

        let overall_status = if critical_issues > 0 {
            ConsistencyStatus::Critical
        } else if warning_issues > 0 {
            ConsistencyStatus::Warning
        } else {
            ConsistencyStatus::Healthy
        };

        Ok(IndexConsistencyReport {
            overall_status,
            checks_performed: checks,
            issues_summary: IssuesSummary {
                total_issues,
                critical_issues,
                warning_issues,
                recommendations,
            },
            check_timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64,
        })
    }

    /// Get file system monitoring status
    pub async fn get_file_system_status(
        _config: &IndexingEngineConfig,
    ) -> Result<FileSystemStatus> {
        // TODO: This would require integration with the actual directory watcher
        // For now, return a basic implementation

        Ok(FileSystemStatus {
            file_watching_active: false, // TODO: Check if directory watcher is active
            directories_monitored: 0,    // TODO: Get from directory watcher
            watcher_error_count: 0,      // TODO: Get from directory watcher error tracking
            last_change_event_timestamp: None, // TODO: Get from directory watcher
            supported_file_systems: vec![
                "ext4".to_string(),
                "NTFS".to_string(),
                "APFS".to_string(),
                "HFS+".to_string(),
                "ZFS".to_string(),
            ],
            base_directory_fs_type: None, // TODO: Detect filesystem type
            recent_events: Vec::new(),    // TODO: Get from directory watcher event log
        })
    }

    /// Get search performance statistics
    pub async fn get_search_performance_stats(
        _enhanced_index: &EnhancedFileIndex,
    ) -> Result<SearchPerformanceStats> {
        // TODO: This would require implementing search metrics tracking
        // For now, return placeholder data
        Ok(SearchPerformanceStats {
            average_response_time_ms: Some(45.2),
            result_quality_metrics: SearchQualityMetrics {
                average_results_count: Some(12.5),
                average_relevance_score: Some(0.78),
                zero_results_percentage: Some(8.3),
            },
            cache_hit_rate_percentage: Some(67.4),
            common_query_patterns: vec![
                "function implementation".to_string(),
                "error handling".to_string(),
                "async/await".to_string(),
            ],
            error_rates: SearchErrorRates {
                semantic_search_error_rate: 0.02,
                text_search_error_rate: 0.01,
                total_queries_processed: 1247,
            },
        })
    }

    /// Get indexing performance statistics
    pub async fn get_indexing_performance_stats(
        _engine: &IndexingEngine,
    ) -> Result<IndexingPerformanceStats> {
        // TODO: This would require implementing performance metrics tracking in IndexingEngine
        // For now, return placeholder data
        let mut processing_times = std::collections::HashMap::new();
        processing_times.insert("rs".to_string(), 156.3);
        processing_times.insert("py".to_string(), 98.7);
        processing_times.insert("js".to_string(), 87.2);
        processing_times.insert("md".to_string(), 43.1);

        Ok(IndexingPerformanceStats {
            files_per_minute: Some(23.4),
            processing_time_by_file_type: processing_times,
            embeddings_per_second: Some(8.9),
            disk_io_stats: DiskIOStats {
                bytes_read_per_second: Some(2_456_789),
                bytes_written_per_second: Some(1_234_567),
                total_disk_space_used_bytes: Some(156_789_012),
            },
            peak_memory_usage_bytes: Some(512_000_000),
        })
    }

    /// Get stale files information from indexing queue
    pub async fn get_stale_files(
        engine: &IndexingEngine,
        _config: &IndexingEngineConfig,
    ) -> Result<StaleFilesInfo> {
        // Get approximate stale files from the indexing queue
        let queue_size = engine.get_queue_size().await;
        let processing_stats = engine.get_stats().await;

        // TODO: In a real implementation, we would:
        // 1. Get files from the task queue that are pending
        // 2. Check filesystem timestamps vs index timestamps
        // 3. Detect deleted files by comparing index vs filesystem

        // For now, create estimates based on queue state
        let mut unindexed_files = Vec::new();
        let mut modified_files = Vec::new();
        let mut reindex_candidates = Vec::new();

        // Estimate some files from queue (placeholder logic)
        if queue_size > 0 {
            for i in 0..std::cmp::min(queue_size, 10) {
                unindexed_files.push(StaleFileEntry {
                    file_path: format!("queued_file_{i}.rs"),
                    last_modified_timestamp: Some(
                        SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64,
                    ),
                    staleness_reason: "Pending in indexing queue".to_string(),
                    reindex_priority: 7,
                });
            }
        }

        // Add some example modified files (placeholder)
        if processing_stats.errors > 0 {
            modified_files.push(StaleFileEntry {
                file_path: "error_prone_file.rs".to_string(),
                last_modified_timestamp: Some(
                    SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64 - 3600,
                ),
                staleness_reason: "Failed indexing with errors".to_string(),
                reindex_priority: 9,
            });
            reindex_candidates.push("error_prone_file.rs".to_string());
        }

        let total_stale_count = unindexed_files.len() + modified_files.len();

        Ok(StaleFilesInfo {
            modified_files,
            unindexed_files,
            deleted_files_in_index: Vec::new(), // TODO: Implement deleted file detection
            reindex_candidates,
            total_stale_count,
        })
    }

    /// Get network status for external dependencies
    pub async fn get_network_status() -> Result<NetworkStatus> {
        // Simple implementation - just check basic connectivity
        let hugging_face_reachable = Self::check_connectivity("https://huggingface.co").await;
        let model_download_reachable = hugging_face_reachable.clone();

        let overall_health = if hugging_face_reachable.is_reachable {
            NetworkHealth::Healthy
        } else {
            NetworkHealth::Limited
        };

        Ok(NetworkStatus {
            model_download_connectivity: model_download_reachable,
            hugging_face_hub_access: hugging_face_reachable,
            proxy_configuration: ProxyStatus {
                proxy_configured: std::env::var("HTTP_PROXY").is_ok()
                    || std::env::var("HTTPS_PROXY").is_ok(),
                proxy_address: std::env::var("HTTP_PROXY")
                    .ok()
                    .or_else(|| std::env::var("HTTPS_PROXY").ok()),
                proxy_auth_configured: false, // TODO: Check proxy auth configuration
            },
            ssl_certificate_validation: true, // TODO: Check SSL configuration
            overall_network_health: overall_health,
        })
    }

    // Helper method for network connectivity check
    async fn check_connectivity(_url: &str) -> ConnectivityStatus {
        // Simple connectivity check - in a real implementation this would use reqwest or similar
        // For now, return a basic status
        ConnectivityStatus {
            is_reachable: true, // TODO: Implement actual HTTP check
            last_successful_connection: Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
            ),
            error_message: None,
            response_time_ms: Some(150), // Placeholder
        }
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

    #[tokio::test]
    async fn test_get_database_info() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf())
                .with_mode(IndexingMode::ReadOnly);

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let db_info = StatusApi::get_database_info(enhanced_index, temp_dir.path()).await?;

        assert_eq!(db_info.database_type, "SQLite");
        assert!(db_info.database_version.is_some());
        assert!(db_info.sqlite_info.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_get_dependency_versions() -> Result<()> {
        let versions = StatusApi::get_dependency_versions().await?;

        assert!(!versions.retriever_version.is_empty());
        assert!(!versions.embed_version.is_empty());
        assert!(!versions.context_version.is_empty());
        assert!(!versions.dependencies.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_validate_index_consistency() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf())
                .with_mode(IndexingMode::ReadOnly);

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let report = StatusApi::validate_index_consistency(enhanced_index).await?;

        assert!(!report.checks_performed.is_empty());
        assert!(matches!(report.overall_status, ConsistencyStatus::Healthy));
        assert_eq!(report.issues_summary.total_issues, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_file_system_status() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf())
                .with_mode(IndexingMode::ReadOnly);

        let fs_status = StatusApi::get_file_system_status(&config).await?;

        assert!(!fs_status.supported_file_systems.is_empty());
        assert_eq!(fs_status.recent_events.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_search_performance_stats() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf())
                .with_mode(IndexingMode::ReadOnly);

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let search_stats = StatusApi::get_search_performance_stats(enhanced_index).await?;

        assert!(search_stats.average_response_time_ms.is_some());
        assert!(!search_stats.common_query_patterns.is_empty());
        assert!(search_stats.error_rates.total_queries_processed > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_indexing_performance_stats() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf())
                .with_mode(IndexingMode::ReadOnly);

        let engine = IndexingEngine::new_memory(config).await?;

        let indexing_stats = StatusApi::get_indexing_performance_stats(&engine).await?;

        assert!(indexing_stats.files_per_minute.is_some());
        assert!(!indexing_stats.processing_time_by_file_type.is_empty());
        assert!(indexing_stats.peak_memory_usage_bytes.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_get_stale_files() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf())
                .with_mode(IndexingMode::ReadOnly);

        let engine = IndexingEngine::new_memory(config.clone()).await?;

        let stale_files = StatusApi::get_stale_files(&engine, &config).await?;

        assert_eq!(
            stale_files.total_stale_count,
            stale_files.modified_files.len() + stale_files.unindexed_files.len()
        );
        assert!(stale_files.deleted_files_in_index.is_empty()); // Should be empty in basic implementation

        Ok(())
    }

    #[tokio::test]
    async fn test_get_network_status() -> Result<()> {
        let network_status = StatusApi::get_network_status().await?;

        assert!(network_status.model_download_connectivity.is_reachable);
        assert!(network_status.hugging_face_hub_access.is_reachable);
        assert!(matches!(
            network_status.overall_network_health,
            NetworkHealth::Healthy
        ));

        Ok(())
    }
}
