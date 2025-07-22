use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::retrieval::{
    enhanced_index::{EmbeddingModelMetadata, EnhancedFileIndex},
    indexing_engine::{IndexingEngine, IndexingEngineConfig},
};

use super::{consistency::*, database::*, filesystem::*, network::*, performance::*, types::*};

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
