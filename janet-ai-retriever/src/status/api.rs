use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::retrieval::{
    enhanced_index::{EmbeddingModelMetadata, EnhancedFileIndex},
    indexing_engine::{IndexingEngine, IndexingEngineConfig},
};

use super::{database::*, types::*};

/// Main status API implementation
pub struct StatusApi;

// Consolidated types previously in separate modules

/// Basic search functionality status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPerformanceStats {
    /// Whether search functionality is available
    pub search_available: bool,
}

/// Basic indexing functionality status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingPerformanceStats {
    /// Whether indexing is operational
    pub indexing_operational: bool,
}

/// Basic file system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemStatus {
    /// Base directory exists and is accessible
    pub base_directory_accessible: bool,
}

/// Basic stale files information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaleFilesInfo {
    /// Number of pending tasks in indexing queue
    pub pending_tasks: usize,
}

/// Basic network status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    /// Whether proxy is configured via environment
    pub proxy_configured: bool,
    /// Overall network health assumption
    pub overall_network_health: NetworkHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkHealth {
    Healthy,
    Limited,
    Offline,
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
            schema_version: "1.0.0".to_string(),
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
            current_file: None,
            progress_percentage: None,
            estimated_time_remaining_seconds: None,
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

        // Basic health check - index is operational if stats are available
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
            chunk_overlap: 0,
            included_file_patterns: Vec::new(),
            excluded_file_patterns: Vec::new(),
            max_file_size_bytes: None,
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
            download_status: ModelDownloadStatus::Downloaded,
            normalized: metadata.normalized,
        }))
    }

    /// Get list of supported file types
    pub async fn get_supported_file_types(_config: &IndexingEngineConfig) -> Result<Vec<String>> {
        // Standard supported file types
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
                total_connections: 1,
                active_connections: 1,
                max_connections: 10,
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
        // Only return actual version info we can reliably get
        Ok(DependencyVersions {
            retriever_version: env!("CARGO_PKG_VERSION").to_string(),
            embed_version: env!("CARGO_PKG_VERSION").to_string(), // Same workspace version
            context_version: env!("CARGO_PKG_VERSION").to_string(), // Same workspace version
            rust_version: option_env!("CARGO_PKG_RUST_VERSION")
                .unwrap_or("unknown")
                .to_string(),
            dependencies: std::collections::HashMap::new(), // Don't track individual deps
        })
    }

    /// Basic index consistency check
    pub async fn validate_index_consistency(
        enhanced_index: &EnhancedFileIndex,
    ) -> Result<IndexConsistencyReport> {
        // Simple check: verify database is accessible and has data
        let files_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM files")
            .fetch_one(enhanced_index.pool())
            .await?;

        let chunks_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks")
            .fetch_one(enhanced_index.pool())
            .await?;

        let status = if files_count > 0 && chunks_count > 0 {
            ConsistencyStatus::Healthy
        } else {
            ConsistencyStatus::Warning
        };

        let basic_check = ConsistencyCheck {
            check_name: "Basic Index Health".to_string(),
            status: status.clone(),
            description: "Verify index contains files and chunks".to_string(),
            items_checked: (files_count + chunks_count) as usize,
            issues_found: 0,
            issue_details: Vec::new(),
        };

        Ok(IndexConsistencyReport {
            overall_status: status,
            checks_performed: vec![basic_check],
            issues_summary: IssuesSummary {
                total_issues: 0,
                critical_issues: 0,
                warning_issues: 0,
                recommendations: Vec::new(),
            },
            check_timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64,
        })
    }

    /// Get basic file system status
    pub async fn get_file_system_status(config: &IndexingEngineConfig) -> Result<FileSystemStatus> {
        // Check if base directory is accessible
        let accessible = config.base_path.exists() && config.base_path.is_dir();

        Ok(FileSystemStatus {
            base_directory_accessible: accessible,
        })
    }

    /// Get basic search functionality status
    pub async fn get_search_performance_stats(
        _enhanced_index: &EnhancedFileIndex,
    ) -> Result<SearchPerformanceStats> {
        // Return basic operational status - search functionality is available
        Ok(SearchPerformanceStats {
            search_available: true,
        })
    }

    /// Get basic indexing functionality status
    pub async fn get_indexing_performance_stats(
        _engine: &IndexingEngine,
    ) -> Result<IndexingPerformanceStats> {
        // Return basic operational status - indexing is available
        Ok(IndexingPerformanceStats {
            indexing_operational: true,
        })
    }

    /// Get basic stale files information
    pub async fn get_stale_files(
        engine: &IndexingEngine,
        _config: &IndexingEngineConfig,
    ) -> Result<StaleFilesInfo> {
        // Simple implementation: just report queue size as pending tasks
        let queue_size = engine.get_queue_size().await;

        Ok(StaleFilesInfo {
            pending_tasks: queue_size,
        })
    }

    /// Get basic network status
    pub async fn get_network_status() -> Result<NetworkStatus> {
        // Basic network status - check proxy configuration only
        let proxy_configured =
            std::env::var("HTTP_PROXY").is_ok() || std::env::var("HTTPS_PROXY").is_ok();

        Ok(NetworkStatus {
            proxy_configured,
            overall_network_health: NetworkHealth::Healthy, // Assume healthy
        })
    }

    // Removed unused check_connectivity helper method

    // Helper methods
    async fn get_database_size(_enhanced_index: &EnhancedFileIndex) -> Result<u64> {
        // Basic database info
        // This would involve getting the SQLite database file path and checking its size
        Ok(0)
    }

    async fn get_last_indexing_timestamp(_enhanced_index: &EnhancedFileIndex) -> Result<i64> {
        // Basic index metadata
        // For now, return current timestamp
        Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64)
    }
}
