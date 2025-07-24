use crate::ServerConfig;
use crate::tools::{
    self, regex_search::RegexSearchRequest, semantic_search::SemanticSearchRequest,
};
use anyhow::Result;
use janet_ai_retriever::{
    retrieval::{
        enhanced_index::EnhancedFileIndex,
        indexing_engine::{IndexingEngine, IndexingEngineConfig},
    },
    status::StatusApi,
};
use rmcp::{ServerHandler, ServiceExt, model::ServerInfo, tool};
use tokio::io::{stdin, stdout};
use tracing::{info, warn};

/// Janet MCP Server that provides search capabilities across codebases
pub struct JanetMcpServer {
    config: ServerConfig,
    #[allow(dead_code)]
    enhanced_index: Option<EnhancedFileIndex>,
    #[allow(dead_code)]
    indexing_engine: Option<IndexingEngine>,
    indexing_config: Option<IndexingEngineConfig>,
}

impl Clone for JanetMcpServer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            enhanced_index: None,  // Don't clone database connections
            indexing_engine: None, // Don't clone indexing engine
            indexing_config: self.indexing_config.clone(),
        }
    }
}

impl JanetMcpServer {
    /// Create a new Janet MCP server with the given configuration
    pub async fn new(config: ServerConfig) -> Result<Self> {
        info!(
            "Initializing Janet MCP server with root: {:?}",
            config.root_dir
        );

        // Try to initialize retriever components if database exists
        let db_path = config.root_dir.join(".janet-ai.db");
        let (enhanced_index, indexing_engine, indexing_config) = if db_path.exists() {
            match Self::initialize_retriever_components(&config).await {
                Ok((index, engine, config)) => (Some(index), Some(engine), Some(config)),
                Err(e) => {
                    warn!(
                        "Failed to initialize retriever components: {}, status will be limited",
                        e
                    );
                    (None, None, None)
                }
            }
        } else {
            info!("No index database found, retriever status will be unavailable");
            (None, None, None)
        };

        Ok(Self {
            config,
            enhanced_index,
            indexing_engine,
            indexing_config,
        })
    }

    /// Try to initialize retriever components for status reporting
    async fn initialize_retriever_components(
        config: &ServerConfig,
    ) -> Result<(EnhancedFileIndex, IndexingEngine, IndexingEngineConfig)> {
        use janet_ai_retriever::retrieval::indexing_mode::IndexingMode;

        // Create enhanced index
        let enhanced_index = EnhancedFileIndex::open(&config.root_dir).await?;

        // Create indexing configuration
        let indexing_config =
            IndexingEngineConfig::new("local".to_string(), config.root_dir.clone())
                .with_mode(IndexingMode::ReadOnly)
                .with_max_workers(4);

        // Create indexing engine
        let indexing_engine = IndexingEngine::new(indexing_config.clone()).await?;

        Ok((enhanced_index, indexing_engine, indexing_config))
    }

    /// Status tool - comprehensive system status and debugging information
    #[tool(
        description = "Show comprehensive system status including index health, configuration, performance metrics, and troubleshooting information"
    )]
    async fn status(&self) -> String {
        info!("Processing comprehensive status request");

        let mut status = format!(
            "Janet AI MCP Server Status\n\
            ================================\n\
            Server Version: {}\n\
            Root Directory: {:?}\n\
            Working Directory: {:?}\n\n",
            env!("CARGO_PKG_VERSION"),
            self.config.root_dir,
            std::env::current_dir().unwrap_or_else(|_| "<unknown>".into())
        );

        // Use comprehensive diagnostic APIs if available
        if let (Some(enhanced_index), Some(indexing_engine), Some(indexing_config)) = (
            &self.enhanced_index,
            &self.indexing_engine,
            &self.indexing_config,
        ) {
            status.push_str(
                &self
                    .get_comprehensive_status(enhanced_index, indexing_engine, indexing_config)
                    .await,
            );
        } else {
            // Fall back to basic status
            status
                .push_str("⚠ Limited status available - retriever components not initialized\n\n");
            status.push_str(&self.get_basic_status());
        }

        status
    }

    /// Get comprehensive status using all diagnostic APIs
    #[allow(clippy::uninlined_format_args)]
    async fn get_comprehensive_status(
        &self,
        enhanced_index: &EnhancedFileIndex,
        indexing_engine: &IndexingEngine,
        indexing_config: &IndexingEngineConfig,
    ) -> String {
        let mut status = String::new();

        // Index Statistics
        status.push_str("Index Statistics\n");
        status.push_str("================\n");
        match StatusApi::get_index_statistics(enhanced_index).await {
            Ok(stats) => {
                status.push_str(&format!(
                    "Total Files: {}\n\
                    Total Chunks: {}\n\
                    Total Embeddings: {}\n\
                    Models Count: {}\n\
                    Schema Version: {}\n",
                    stats.total_files,
                    stats.total_chunks,
                    stats.total_embeddings,
                    stats.models_count,
                    stats.schema_version
                ));

                if let Some(db_size) = stats.database_size_bytes {
                    status.push_str(&format!(
                        "Database Size: {:.2} MB\n",
                        db_size as f64 / 1_048_576.0
                    ));
                }

                if let Some(last_index) = stats.last_indexing_timestamp {
                    let datetime =
                        std::time::UNIX_EPOCH + std::time::Duration::from_secs(last_index as u64);
                    status.push_str(&format!("Last Indexing: {:?}\n", datetime));
                }
            }
            Err(e) => status.push_str(&format!("⚠ Failed to get index statistics: {}\n", e)),
        }
        status.push('\n');

        // Index Health
        status.push_str("Index Health\n");
        status.push_str("============\n");
        match StatusApi::get_index_health(enhanced_index).await {
            Ok(health) => {
                status.push_str(&format!(
                    "Overall Status: {:?}\n\
                    Database Connected: {}\n\
                    Database Integrity: {}\n\
                    Directory Writable: {}\n",
                    health.overall_status,
                    if health.database_connected {
                        "✓"
                    } else {
                        "✗"
                    },
                    if health.database_integrity_ok {
                        "✓"
                    } else {
                        "✗"
                    },
                    if health.index_directory_writable {
                        "✓"
                    } else {
                        "✗"
                    }
                ));

                if let Some(error) = &health.database_error {
                    status.push_str(&format!("Database Error: {}\n", error));
                }

                if let Some(disk_space) = health.available_disk_space_bytes {
                    status.push_str(&format!(
                        "Available Disk Space: {:.2} GB\n",
                        disk_space as f64 / 1_073_741_824.0
                    ));
                }

                if let Some(memory) = health.estimated_memory_usage_bytes {
                    status.push_str(&format!(
                        "Estimated Memory Usage: {:.2} MB\n",
                        memory as f64 / 1_048_576.0
                    ));
                }
            }
            Err(e) => status.push_str(&format!("⚠ Failed to get index health: {}\n", e)),
        }
        status.push('\n');

        // Indexing Status
        status.push_str("Indexing Status\n");
        status.push_str("===============\n");
        match StatusApi::get_indexing_status(indexing_engine).await {
            Ok(indexing_status) => {
                status.push_str(&format!(
                    "Is Running: {}\n\
                    Queue Size: {}\n\
                    Files Processed: {}\n\
                    Chunks Created: {}\n\
                    Embeddings Generated: {}\n\
                    Error Count: {}\n",
                    if indexing_status.is_running {
                        "✓"
                    } else {
                        "✗"
                    },
                    indexing_status.queue_size,
                    indexing_status.files_processed,
                    indexing_status.chunks_created,
                    indexing_status.embeddings_generated,
                    indexing_status.error_count
                ));

                if let Some(current_file) = &indexing_status.current_file {
                    status.push_str(&format!("Current File: {}\n", current_file));
                }

                if let Some(progress) = indexing_status.progress_percentage {
                    status.push_str(&format!("Progress: {:.1}%\n", progress));
                }

                if let Some(eta) = indexing_status.estimated_time_remaining_seconds {
                    status.push_str(&format!("ETA: {} seconds\n", eta));
                }
            }
            Err(e) => status.push_str(&format!("⚠ Failed to get indexing status: {}\n", e)),
        }
        status.push('\n');

        // Configuration
        status.push_str("Configuration\n");
        status.push_str("=============\n");
        match StatusApi::get_indexing_config(indexing_config).await {
            Ok(config) => {
                status.push_str(&format!(
                    "Repository: {}\n\
                    Base Path: {}\n\
                    Indexing Mode: {}\n\
                    Max Chunk Size: {}\n\
                    Chunk Overlap: {}\n\
                    Worker Threads: {}\n",
                    config.repository,
                    config.base_path,
                    config.indexing_mode,
                    config.max_chunk_size,
                    config.chunk_overlap,
                    config.worker_thread_count
                ));

                if let Some(max_file_size) = config.max_file_size_bytes {
                    status.push_str(&format!(
                        "Max File Size: {:.2} MB\n",
                        max_file_size as f64 / 1_048_576.0
                    ));
                }

                if !config.included_file_patterns.is_empty() {
                    status.push_str(&format!(
                        "Included Patterns: {:?}\n",
                        config.included_file_patterns
                    ));
                }

                if !config.excluded_file_patterns.is_empty() {
                    status.push_str(&format!(
                        "Excluded Patterns: {:?}\n",
                        config.excluded_file_patterns
                    ));
                }
            }
            Err(e) => status.push_str(&format!("⚠ Failed to get indexing config: {}\n", e)),
        }
        status.push('\n');

        // Embedding Model Info
        status.push_str("Embedding Model\n");
        status.push_str("===============\n");
        // Try to get embedding model metadata from enhanced index
        match enhanced_index.get_all_embedding_models().await {
            Ok(models) => {
                if models.is_empty() {
                    status.push_str("No embedding models registered\n");
                } else {
                    for metadata in &models {
                        match StatusApi::get_embedding_model_info(Some(metadata)).await {
                            Ok(Some(model_info)) => {
                                status.push_str(&format!(
                                    "Model Name: {}\n\
                                    Provider: {}\n\
                                    Dimensions: {}\n\
                                    Download Status: {:?}\n\
                                    Normalized: {}\n",
                                    model_info.model_name,
                                    model_info.provider,
                                    model_info.dimensions,
                                    model_info.download_status,
                                    model_info.normalized
                                ));

                                if let Some(file_size) = model_info.model_file_size_bytes {
                                    status.push_str(&format!(
                                        "Model File Size: {:.2} MB\n",
                                        file_size as f64 / 1_048_576.0
                                    ));
                                }

                                if let Some(location) = &model_info.model_file_location {
                                    status.push_str(&format!("Model Location: {}\n", location));
                                }

                                if !model_info.supported_languages.is_empty() {
                                    status.push_str(&format!(
                                        "Supported Languages: {:?}\n",
                                        model_info.supported_languages
                                    ));
                                }

                                if let Some(onnx_info) = &model_info.onnx_runtime_info {
                                    status.push_str(&format!(
                                        "GPU Available: {}\n",
                                        if onnx_info.gpu_available {
                                            "✓"
                                        } else {
                                            "✗"
                                        }
                                    ));

                                    if let Some(gpu_device) = &onnx_info.gpu_device {
                                        status.push_str(&format!("GPU Device: {}\n", gpu_device));
                                    }

                                    if let Some(runtime_version) = &onnx_info.runtime_version {
                                        status.push_str(&format!(
                                            "ONNX Runtime Version: {}\n",
                                            runtime_version
                                        ));
                                    }
                                }
                                status.push('\n');
                            }
                            Ok(None) => {
                                status.push_str("No embedding model information available\n")
                            }
                            Err(e) => status.push_str(&format!(
                                "⚠ Failed to get embedding model info: {}\n",
                                e
                            )),
                        }
                    }
                }
            }
            Err(e) => status.push_str(&format!("⚠ Failed to get embedding models: {}\n", e)),
        }
        status.push('\n');

        // Database Information
        status.push_str("Database Information\n");
        status.push_str("====================\n");
        match StatusApi::get_database_info(enhanced_index, &self.config.root_dir).await {
            Ok(db_info) => {
                status.push_str(&format!("Database Type: {}\n", db_info.database_type));

                if let Some(version) = &db_info.database_version {
                    status.push_str(&format!("Database Version: {}\n", version));
                }

                if let Some(total_size) = db_info.total_size_bytes {
                    status.push_str(&format!(
                        "Total Size: {:.2} MB\n",
                        total_size as f64 / 1_048_576.0
                    ));
                }

                status.push_str(&format!(
                    "Connection Pool - Total: {}, Active: {}, Max: {}\n",
                    db_info.connection_pool_status.total_connections,
                    db_info.connection_pool_status.active_connections,
                    db_info.connection_pool_status.max_connections
                ));

                if let Some(timeout) = db_info.connection_pool_status.connection_timeout_seconds {
                    status.push_str(&format!("Connection Timeout: {}s\n", timeout));
                }

                if let Some(sqlite_info) = &db_info.sqlite_info {
                    status.push_str(&format!("SQLite Version: {}\n", sqlite_info.version));

                    if let Some(journal_mode) = &sqlite_info.journal_mode {
                        status.push_str(&format!("Journal Mode: {}\n", journal_mode));
                    }

                    if let Some(sync_mode) = &sqlite_info.synchronous_mode {
                        status.push_str(&format!("Synchronous Mode: {}\n", sync_mode));
                    }

                    if let Some(page_size) = sqlite_info.page_size {
                        status.push_str(&format!("Page Size: {} bytes\n", page_size));
                    }

                    if let Some(page_count) = sqlite_info.page_count {
                        status.push_str(&format!("Page Count: {}\n", page_count));
                    }
                }

                if !db_info.database_files.is_empty() {
                    status.push_str("Database Files:\n");
                    for file in &db_info.database_files {
                        status.push_str(&format!("  {} ({})", file.path, file.file_type));
                        if let Some(size) = file.size_bytes {
                            status.push_str(&format!(" - {:.2} MB", size as f64 / 1_048_576.0));
                        }
                        status.push('\n');
                    }
                }
            }
            Err(e) => status.push_str(&format!("⚠ Failed to get database info: {}\n", e)),
        }
        status.push('\n');

        // Dependency Versions
        status.push_str("Dependency Versions\n");
        status.push_str("===================\n");
        match StatusApi::get_dependency_versions().await {
            Ok(versions) => {
                status.push_str(&format!(
                    "Retriever Version: {}\n\
                    Embed Version: {}\n\
                    Context Version: {}\n\
                    Rust Version: {}\n",
                    versions.retriever_version,
                    versions.embed_version,
                    versions.context_version,
                    versions.rust_version
                ));

                if !versions.dependencies.is_empty() {
                    status.push_str("Key Dependencies:\n");
                    for (name, version) in &versions.dependencies {
                        status.push_str(&format!("  {}: {}\n", name, version));
                    }
                }
            }
            Err(e) => status.push_str(&format!("⚠ Failed to get dependency versions: {}\n", e)),
        }
        status.push('\n');

        // Network Status
        status.push_str("Network Status\n");
        status.push_str("==============\n");
        match StatusApi::get_network_status().await {
            Ok(network_status) => {
                status.push_str(&format!(
                    "Overall Network Health: {:?}\n\
                    SSL Certificate Validation: {}\n",
                    network_status.overall_network_health,
                    if network_status.ssl_certificate_validation {
                        "✓"
                    } else {
                        "✗"
                    }
                ));

                status.push_str(&format!(
                    "Model Download Connectivity: {}\n",
                    if network_status.model_download_connectivity.is_reachable {
                        "✓"
                    } else {
                        "✗"
                    }
                ));

                if let Some(error) = &network_status.model_download_connectivity.error_message {
                    status.push_str(&format!("  Error: {}\n", error));
                }

                if let Some(response_time) =
                    network_status.model_download_connectivity.response_time_ms
                {
                    status.push_str(&format!("  Response Time: {}ms\n", response_time));
                }

                status.push_str(&format!(
                    "Hugging Face Hub Access: {}\n",
                    if network_status.hugging_face_hub_access.is_reachable {
                        "✓"
                    } else {
                        "✗"
                    }
                ));

                if network_status.proxy_configuration.proxy_configured {
                    status.push_str("Proxy: Configured\n");
                    if let Some(proxy_addr) = &network_status.proxy_configuration.proxy_address {
                        status.push_str(&format!("  Address: {}\n", proxy_addr));
                    }
                    status.push_str(&format!(
                        "  Auth: {}\n",
                        if network_status.proxy_configuration.proxy_auth_configured {
                            "✓"
                        } else {
                            "✗"
                        }
                    ));
                } else {
                    status.push_str("Proxy: Not configured\n");
                }
            }
            Err(e) => status.push_str(&format!("⚠ Failed to get network status: {}\n", e)),
        }
        status.push('\n');

        // Search Performance
        status.push_str("Search Performance\n");
        status.push_str("==================\n");
        match StatusApi::get_search_performance_stats(enhanced_index).await {
            Ok(perf_stats) => {
                if let Some(avg_response_time) = perf_stats.average_response_time_ms {
                    status.push_str(&format!(
                        "Average Response Time: {:.1}ms\n",
                        avg_response_time
                    ));
                }

                if let Some(cache_hit_rate) = perf_stats.cache_hit_rate_percentage {
                    status.push_str(&format!("Cache Hit Rate: {:.1}%\n", cache_hit_rate));
                }

                let quality = &perf_stats.result_quality_metrics;
                if let Some(avg_results) = quality.average_results_count {
                    status.push_str(&format!("Average Results Count: {:.1}\n", avg_results));
                }

                if let Some(avg_relevance) = quality.average_relevance_score {
                    status.push_str(&format!("Average Relevance Score: {:.2}\n", avg_relevance));
                }

                if let Some(zero_results) = quality.zero_results_percentage {
                    status.push_str(&format!("Zero Results Rate: {:.1}%\n", zero_results));
                }

                status.push_str(&format!(
                    "Total Queries Processed: {}\n\
                    Semantic Search Error Rate: {:.2}%\n\
                    Text Search Error Rate: {:.2}%\n",
                    perf_stats.error_rates.total_queries_processed,
                    perf_stats.error_rates.semantic_search_error_rate * 100.0,
                    perf_stats.error_rates.text_search_error_rate * 100.0
                ));

                if !perf_stats.common_query_patterns.is_empty() {
                    status.push_str("Common Query Patterns:\n");
                    for pattern in &perf_stats.common_query_patterns {
                        status.push_str(&format!("  - {}\n", pattern));
                    }
                }
            }
            Err(e) => status.push_str(&format!(
                "⚠ Failed to get search performance stats: {}\n",
                e
            )),
        }
        status.push('\n');

        status
    }

    /// Get basic status when retriever components are not available
    fn get_basic_status(&self) -> String {
        let mut status = String::new();

        // Index Infrastructure Status
        status.push_str(&self.get_index_status());

        // Search Capabilities Status
        status.push_str(&self.get_search_status());

        // System Resource Status
        status.push_str(&self.get_system_status());

        // Troubleshooting Information
        status.push_str(&self.get_troubleshooting_info());

        status
    }

    /// Get index infrastructure status
    fn get_index_status(&self) -> String {
        let index_db_path = self.config.root_dir.join(".janet-ai.db");
        let index_exists = index_db_path.exists();

        let mut status = String::from(
            "Index Infrastructure\n\
            --------------------\n",
        );

        status.push_str(&format!(
            "Index Database: {}\n\
            Location: {:?}\n",
            if index_exists {
                "✓ Found"
            } else {
                "✗ Missing"
            },
            index_db_path
        ));

        if index_exists {
            if let Ok(metadata) = std::fs::metadata(&index_db_path) {
                status.push_str(&format!(
                    "Database Size: {:.2} MB\n\
                    Last Modified: {:?}\n",
                    metadata.len() as f64 / 1_048_576.0,
                    metadata
                        .modified()
                        .map(|t| format!("{t:?}"))
                        .unwrap_or_else(|_| "Unknown".to_string())
                ));
            }
        }

        // Check index directory permissions
        let index_dir = &self.config.root_dir;
        let dir_status = if index_dir.exists() {
            if index_dir
                .metadata()
                .map(|m| !m.permissions().readonly())
                .unwrap_or(false)
            {
                "✓ Writable"
            } else {
                "⚠ Read-only"
            }
        } else {
            "✗ Missing"
        };

        status.push_str(&format!("Index Directory: {dir_status}\n"));

        // Check embedding model availability (simplified check)
        let embedding_status = "⚠ Unknown (requires janet-ai-retriever integration)";
        status.push_str(&format!("Embedding Model: {embedding_status}\n\n"));

        status
    }

    /// Get search capabilities status
    fn get_search_status(&self) -> String {
        let index_db_path = self.config.root_dir.join(".janet-ai.db");
        let index_available = index_db_path.exists();

        format!(
            "Search Capabilities\n\
            -------------------\n\
            Regex Search: ✓ Always available\n\
            Semantic Search: {}\n\
            File Content Search: {}\n\n",
            if index_available {
                "✓ Available"
            } else {
                "⚠ Requires indexing"
            },
            if index_available {
                "✓ Available"
            } else {
                "⚠ Requires indexing"
            }
        )
    }

    /// Get system resource status
    fn get_system_status(&self) -> String {
        let mut status = String::from(
            "System Resources\n\
            ----------------\n",
        );

        // Check root directory accessibility
        let root_accessible = self.config.root_dir.exists()
            && self
                .config
                .root_dir
                .metadata()
                .map(|m| m.is_dir())
                .unwrap_or(false);

        status.push_str(&format!(
            "Root Directory Access: {}\n",
            if root_accessible {
                "✓ Accessible"
            } else {
                "✗ Inaccessible"
            }
        ));

        // Check available disk space (simplified)
        if let Ok(root_metadata) = std::fs::metadata(&self.config.root_dir) {
            status.push_str(&format!(
                "Root Directory Type: {}\n",
                if root_metadata.is_dir() {
                    "Directory"
                } else {
                    "File"
                }
            ));
        }

        // Runtime information
        status.push_str(&format!("Platform: {}\n\n", std::env::consts::OS));

        status
    }

    /// Get troubleshooting information
    fn get_troubleshooting_info(&self) -> String {
        let index_db_path = self.config.root_dir.join(".janet-ai.db");
        let index_exists = index_db_path.exists();

        let mut info = String::from(
            "Troubleshooting\n\
            ---------------\n",
        );

        if !index_exists {
            info.push_str(
                "⚠ No index found. To enable semantic search:\n\
                1. Install janet-ai-retriever: cargo install janet-ai-retriever\n\
                2. Index your repository: janet-ai-retriever index --repo .\n\
                3. Restart the MCP server\n\n",
            );
        } else {
            info.push_str(
                "✓ Index database found\n\
                • Semantic search should be available\n\
                • If search fails, check embedding model setup\n\n",
            );
        }

        info.push_str(
            "Common Issues:\n\
            • Slow search: Check index freshness with 'janet-ai-retriever status'\n\
            • No results: Verify files are indexed and embeddings generated\n\
            • Permission errors: Ensure write access to .janet directory\n\
            • Memory issues: Reduce batch size in embedding configuration\n\n\
            \
            For detailed diagnostics: Check TODO.md for additional status functions\n\
            Log files: Check stdout/stderr for detailed error messages\n",
        );

        info
    }

    /// Regex search tool - search project files, dependencies, and docs
    #[tool(
        description = "Search files using regex patterns across project files, dependencies, and autogenerated docs"
    )]
    async fn regex_search(&self, request: RegexSearchRequest) -> Result<String, String> {
        tools::regex_search::regex_search(&self.config, request).await
    }

    /// Semantic search tool - search using embeddings
    #[tool(description = "Search files using semantic similarity with embeddings")]
    async fn semantic_search(&self, request: SemanticSearchRequest) -> Result<String, String> {
        tools::semantic_search::semantic_search(&self.config, request).await
    }

    /// Serve the MCP server using stdio transport
    pub async fn serve_stdio(&self) -> Result<()> {
        info!("Starting MCP server with stdio transport");

        let transport = (stdin(), stdout());
        let server = self.clone().serve(transport).await?;
        let quit_reason = server.waiting().await?;

        info!("MCP server quit: {:?}", quit_reason);
        Ok(())
    }
}

impl ServerHandler for JanetMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Janet AI MCP Server - provides regex and semantic search capabilities across codebases".into()),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_creation() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        let server = JanetMcpServer::new(config).await;
        assert!(server.is_ok(), "Server creation should succeed");
    }

    #[tokio::test]
    async fn test_status_without_index() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        let server = JanetMcpServer::new(config)
            .await
            .expect("Server creation should succeed");
        let status_output = server.status().await;

        // Should show basic status since no .janet-ai.db exists
        assert!(status_output.contains("Janet AI MCP Server Status"));
        assert!(status_output.contains("Server Version:"));
        assert!(status_output.contains("Root Directory:"));
        assert!(status_output.contains("Working Directory:"));
        assert!(
            status_output
                .contains("⚠ Limited status available - retriever components not initialized")
        );

        // Should include basic status sections
        assert!(status_output.contains("Index Infrastructure"));
        assert!(status_output.contains("Search Capabilities"));
        assert!(status_output.contains("System Resources"));
        assert!(status_output.contains("Troubleshooting"));
    }

    #[test]
    fn test_basic_status_sections() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        // Create a mock server (without async initialization) for testing basic status
        let server = JanetMcpServer {
            config: config.clone(),
            enhanced_index: None,
            indexing_engine: None,
            indexing_config: None,
        };

        // Test individual status sections
        let index_status = server.get_index_status();
        assert!(index_status.contains("Index Infrastructure"));
        assert!(index_status.contains("Index Database:"));
        assert!(index_status.contains("✗ Missing")); // No database should exist

        let search_status = server.get_search_status();
        assert!(search_status.contains("Search Capabilities"));
        assert!(search_status.contains("Regex Search: ✓ Always available"));
        assert!(search_status.contains("Semantic Search:"));
        assert!(search_status.contains("⚠ Requires indexing")); // No database exists

        let system_status = server.get_system_status();
        assert!(system_status.contains("System Resources"));
        assert!(system_status.contains("Root Directory Access:"));
        assert!(system_status.contains("Platform:"));

        let troubleshooting_info = server.get_troubleshooting_info();
        assert!(troubleshooting_info.contains("Troubleshooting"));
        assert!(troubleshooting_info.contains("⚠ No index found"));
        assert!(troubleshooting_info.contains("Common Issues:"));
    }

    #[tokio::test]
    async fn test_status_with_existing_database_file() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");

        // Create an empty .janet-ai.db file to simulate existing database
        let db_path = temp_dir.path().join(".janet-ai.db");
        std::fs::write(&db_path, b"fake database content").expect("Failed to create fake database");

        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        let server = JanetMcpServer::new(config)
            .await
            .expect("Server creation should succeed");
        let status_output = server.status().await;

        // Should attempt to use comprehensive status since database file exists,
        // but fall back to basic status due to initialization failure
        assert!(status_output.contains("Janet AI MCP Server Status"));

        // Could contain either comprehensive status (if initialization succeeds)
        // or limited status with basic sections (if initialization fails)
        assert!(
            status_output.contains("⚠ Limited status available")
                || status_output.contains("Index Statistics")
                || status_output.contains("Index Infrastructure")
        );
    }

    #[test]
    fn test_server_clone() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        let server = JanetMcpServer {
            config: config.clone(),
            enhanced_index: None,
            indexing_engine: None,
            indexing_config: Some(
                janet_ai_retriever::retrieval::indexing_engine::IndexingEngineConfig::new(
                    "test".to_string(),
                    temp_dir.path().to_path_buf(),
                ),
            ),
        };

        let cloned_server = server.clone();

        // Should clone config and indexing_config but not the components
        assert_eq!(cloned_server.config.root_dir, server.config.root_dir);
        assert!(cloned_server.enhanced_index.is_none());
        assert!(cloned_server.indexing_engine.is_none());
        assert!(cloned_server.indexing_config.is_some());
    }

    #[tokio::test]
    async fn test_initialize_retriever_components_creates_database() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        let db_path = temp_dir.path().join(".janet-ai.db");
        assert!(!db_path.exists(), "Database should not exist initially");

        // Should succeed and create database
        let result = JanetMcpServer::initialize_retriever_components(&config).await;
        assert!(
            result.is_ok(),
            "Should succeed and create database if missing"
        );

        // Database should now exist
        assert!(db_path.exists(), "Database should be created");
    }
}
