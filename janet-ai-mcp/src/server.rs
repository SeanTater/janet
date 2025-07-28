#![allow(dead_code)]
use std::sync::Arc;

// Much of the code is used by the MCP server tools, which foul the static analysis
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
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::tool::ToolRouter,
    model::{CallToolResult, Content, ServerInfo},
};
use tokio::io::{stdin, stdout};
use tokio::sync::Mutex;
use tracing::info;

/// Janet MCP Server that provides search capabilities across codebases.
///
/// This server implements the Model Context Protocol (MCP) to provide AI tools
/// with advanced search capabilities over indexed codebases. It combines:
/// - **Regex search**: Fast pattern-based text search with context
/// - **Semantic search**: AI-powered similarity search using embeddings
/// - **Status reporting**: Comprehensive system health and configuration info
///
/// The server requires a pre-built index database created by janet-ai-retriever.
/// It communicates with MCP clients (like Claude Desktop) over stdio using
/// the standard MCP protocol.
///
/// # Architecture
/// ```text
/// MCP Client (Claude) ↔ stdio ↔ JanetMcpServer ↔ IndexingEngine ↔ SQLite DB
///                                        ↓
///                                 EnhancedFileIndex
/// ```
#[derive(Clone, Debug)]
pub struct JanetMcpServer {
    config: ServerConfig,
    enhanced_index: EnhancedFileIndex,
    indexing_engine: Arc<Mutex<IndexingEngine>>,
    indexing_config: IndexingEngineConfig,
    tool_router: ToolRouter<Self>,
}

#[rmcp::tool_router]
impl JanetMcpServer {
    /// Create a new Janet MCP server with the given configuration.
    ///
    /// This initializes the server with all necessary components for providing
    /// search capabilities. The server requires an existing `.janet-ai.db` index
    /// file in the root directory - use janet-ai-retriever to create this first.
    ///
    /// # Arguments
    /// * `config` - Server configuration specifying the root directory
    ///
    /// # Returns
    /// A new JanetMcpServer instance ready to serve MCP requests
    ///
    /// # Errors
    /// - If the `.janet-ai.db` index file is not found in the root directory
    /// - Database connection or initialization errors
    /// - IndexingEngine initialization errors
    ///
    /// # Example
    pub async fn new(config: ServerConfig) -> Result<Self> {
        info!(
            "Initializing Janet MCP server with root: {:?}",
            config.root_dir
        );

        let db_path = config.root_dir.join(".janet-ai.db");
        if !db_path.exists() {
            return Err(anyhow::anyhow!(
                "No index database found at {:?}. Run 'janet-ai-retriever index --repo .' to create one.",
                db_path
            ));
        }

        // Initialize retriever components
        let enhanced_index = EnhancedFileIndex::open(&config.root_dir).await?;
        let indexing_config =
            IndexingEngineConfig::new("local".to_string(), config.root_dir.clone())
                .with_max_workers(4);
        let mut indexing_engine = IndexingEngine::new(indexing_config.clone()).await?;

        // Start the indexing engine with full reindex to populate the database
        indexing_engine.start(true).await?;

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

        // Get comprehensive status from StatusApi
        let status_result = StatusApi::get_comprehensive_status(
            &self.enhanced_index,
            &*self.indexing_engine.lock().await,
            &self.indexing_config,
            &self.config.root_dir,
        )
        .await
        .map_err(|err| McpError::internal_error(format!("Failed to get status: {err}"), None))?
        .to_toml()
        .map_err(|err| {
            McpError::internal_error(format!("Failed to serialize status to TOML: {err}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(status_result)]))
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
    /// Start serving MCP requests over stdio transport.
    ///
    /// This method starts the MCP server and begins listening for requests over
    /// standard input/output. This is the standard transport mechanism for MCP
    /// servers when integrated with clients like Claude Desktop.
    ///
    /// The server will continue running until it receives a shutdown signal or
    /// encounters an error. All MCP protocol messages are handled automatically.
    ///
    /// # Returns
    /// `Ok(())` when the server shuts down gracefully
    ///
    /// # Errors
    /// - MCP protocol communication errors
    /// - Stdin/stdout transport errors
    /// - Tool execution errors
    ///
    /// # Example
    pub async fn serve_stdio(self) -> Result<()> {
        info!("Starting MCP server with stdio transport");

        let transport = (stdin(), stdout());
        let server = self.serve(transport).await?;
        let quit_reason = server.waiting().await?;

        info!("MCP server quit: {:?}", quit_reason);
        Ok(())
    }
}

#[rmcp::tool_handler]
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
    async fn test_server_creation_requires_index() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        // Should fail without index database
        let result = JanetMcpServer::new(config).await;
        assert!(
            result.is_err(),
            "Server creation should fail without index database"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No index database found")
        );
    }

    #[tokio::test]
    async fn test_enhanced_file_index_creates_database() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        let db_path = temp_dir.path().join(".janet-ai.db");
        assert!(!db_path.exists(), "Database should not exist initially");

        // Should succeed and create database
        let result = EnhancedFileIndex::open(&config.root_dir).await;
        assert!(
            result.is_ok(),
            "Should succeed and create database if missing"
        );

        // Database should now exist
        assert!(db_path.exists(), "Database should be created");
    }
}
