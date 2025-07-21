use crate::ServerConfig;
use crate::tools::{
    self, regex_search::RegexSearchRequest, semantic_search::SemanticSearchRequest,
};
use anyhow::Result;
use rmcp::{ServerHandler, ServiceExt, model::ServerInfo, tool};
use tokio::io::{stdin, stdout};
use tracing::info;

/// Janet MCP Server that provides search capabilities across codebases
#[derive(Clone)]
pub struct JanetMcpServer {
    #[allow(dead_code)]
    config: ServerConfig,
}

impl JanetMcpServer {
    /// Create a new Janet MCP server with the given configuration
    pub async fn new(config: ServerConfig) -> Result<Self> {
        info!(
            "Initializing Janet MCP server with root: {:?}",
            config.root_dir
        );

        Ok(Self { config })
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
        let index_db_path = self.config.root_dir.join(".janet.db");
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
        let index_db_path = self.config.root_dir.join(".janet.db");
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
        let index_db_path = self.config.root_dir.join(".janet.db");
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
        eprintln!("DEBUG: serve_stdio called");

        let transport = (stdin(), stdout());
        eprintln!("DEBUG: Created stdio transport");

        let server = self.clone().serve(transport).await?;
        eprintln!("DEBUG: Server.serve() completed, waiting for connections");

        let quit_reason = server.waiting().await?;
        eprintln!("DEBUG: Server quit with reason: {quit_reason:?}");

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
