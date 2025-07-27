use std::sync::Arc;

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
use rmcp::handler::server::tool::Parameters;
use rmcp::model::{Implementation, ProtocolVersion, ServerCapabilities};
use rmcp::tool_handler;
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::tool::ToolRouter,
    model::{CallToolResult, Content, ServerInfo},
};
use tokio::io::{stdin, stdout};
use tokio::sync::Mutex;
use tracing::info;

/// Janet MCP Server that provides search capabilities across codebases
#[derive(Clone)]
pub struct JanetMcpServer {
    config: ServerConfig,
    #[allow(dead_code)]
    enhanced_index: EnhancedFileIndex,
    #[allow(dead_code)]
    indexing_engine: Arc<Mutex<IndexingEngine>>,
    indexing_config: IndexingEngineConfig,
    tool_router: ToolRouter<Self>,
}

#[rmcp::tool_router]
impl JanetMcpServer {
    /// Create a new Janet MCP server with the given configuration
    pub async fn new(config: ServerConfig) -> Result<Self> {
        info!(
            "Initializing Janet MCP server with root: {:?}",
            config.root_dir
        );

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

        Ok(Self {
            config,
            enhanced_index,
            indexing_engine: Arc::new(Mutex::new(indexing_engine)),
            indexing_config,
            tool_router: Self::tool_router(),
        })
    }

    /// Status tool - comprehensive system status and debugging information
    #[rmcp::tool(
        description = "Show comprehensive system status including index health, configuration, performance metrics, and troubleshooting information"
    )]
    async fn status(&self) -> Result<CallToolResult, McpError> {
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
        status.push_str(&self.get_comprehensive_status().await);

        Ok(CallToolResult::success(vec![Content::text(status)]))
    }

    /// Get comprehensive status using all diagnostic APIs
    #[allow(clippy::uninlined_format_args)]
    async fn get_comprehensive_status(&self) -> String {
        let mut status = String::new();

        // Index Statistics
        status.push_str("Index Statistics\n");
        status.push_str("================\n");
        match StatusApi::get_index_statistics(&self.enhanced_index).await {
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
        match StatusApi::get_index_health(&self.enhanced_index).await {
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
        let indexing_engine = self.indexing_engine.lock().await;
        match StatusApi::get_indexing_status(&indexing_engine).await {
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
        match StatusApi::get_indexing_config(&self.indexing_config).await {
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
        match self.enhanced_index.get_all_embedding_models().await {
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

                                // Simplified model info - removed untracked fields
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
        match StatusApi::get_database_info(&self.enhanced_index, &self.config.root_dir).await {
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
                    Rust Version: {}\n",
                    versions.retriever_version, versions.rust_version
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
                    "Overall Network Health: {:?}\n",
                    network_status.overall_network_health
                ));

                status.push_str(&format!(
                    "Proxy: {}\n",
                    if network_status.proxy_configured {
                        "Configured"
                    } else {
                        "Not configured"
                    }
                ));
            }
            Err(e) => status.push_str(&format!("⚠ Failed to get network status: {}\n", e)),
        }
        status.push('\n');

        // Search Performance
        status.push_str("Search Performance\n");
        status.push_str("==================\n");
        match StatusApi::get_search_performance_stats(&self.enhanced_index).await {
            Ok(perf_stats) => {
                status.push_str(&format!(
                    "Search Available: {}\n",
                    if perf_stats.search_available {
                        "✓"
                    } else {
                        "✗"
                    }
                ));
            }
            Err(e) => status.push_str(&format!(
                "⚠ Failed to get search performance stats: {}\n",
                e
            )),
        }
        status.push('\n');

        status
    }

    /// Regex search tool - search project files, dependencies, and docs
    #[rmcp::tool(
        description = "Search files using regex patterns across project files, dependencies, and autogenerated docs"
    )]
    async fn regex_search(
        &self,
        Parameters(request): Parameters<RegexSearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        match tools::regex_search::regex_search(&self.config, request).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result)])),
            Err(err) => Err(McpError::internal_error(err, None)),
        }
    }

    /// Semantic search tool - search using embeddings
    #[rmcp::tool(description = "Search files using semantic similarity with embeddings")]
    async fn semantic_search(
        &self,
        Parameters(request): Parameters<SemanticSearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        match tools::semantic_search::semantic_search(&self.config, request).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result)])),
            Err(err) => Err(McpError::internal_error(err, None)),
        }
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

#[tool_handler]
impl ServerHandler for JanetMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::LATEST,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("Janet AI MCP Server - provides regex and semantic search capabilities across codebases".into()),
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
    async fn test_initialize_retriever_components_creates_database() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        let db_path = temp_dir.path().join(".janet-ai.db");
        assert!(!db_path.exists(), "Database should not exist initially");

        // Should succeed and create database
        let result = JanetMcpServer::new(config).await;
        assert!(
            result.is_ok(),
            "Should succeed and create database if missing"
        );

        // Database should now exist
        assert!(db_path.exists(), "Database should be created");
    }
}
