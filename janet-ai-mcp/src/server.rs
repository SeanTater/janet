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
        match StatusApi::get_comprehensive_status(
            enhanced_index,
            indexing_engine,
            indexing_config,
            &self.config.root_dir,
        )
        .await
        {
            Ok(status) => match status.to_toml() {
                Ok(toml_output) => toml_output,
                Err(e) => format!("⚠ Failed to serialize status to TOML: {}", e),
            },
            Err(e) => format!("⚠ Failed to get comprehensive status: {}", e),
        }
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
