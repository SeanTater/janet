//! # janet-ai-mcp
//!
//! A Model Context Protocol (MCP) server for janet-ai that provides semantic and regex
//! search capabilities across codebases. This crate implements an MCP server that AI
//! clients (like Claude Desktop) can use to search and understand codebases.
//!
//! ## Features
//!
//! - **Regex Search**: Fast pattern-based text search with surrounding context
//! - **Semantic Search**: AI-powered similarity search using vector embeddings
//! - **System Status**: Comprehensive health monitoring and diagnostics
//! - **MCP Integration**: Standard Model Context Protocol implementation
//!
//! ## Architecture
//!
//! The server integrates multiple janet-ai components:
//! - [`janet-ai-retriever`] for file indexing and database management
//! - [`janet-ai-embed`] for generating semantic embeddings
//! - [`janet-ai-context`] for intelligent text chunking
//!
//! ## Quick Start
//!
//! ### 1. Index your codebase
//! ```bash
//! # First, create an index of your codebase
//! janet-ai-retriever index --repo .
//! ```
//!
//! ### 2. Start the MCP server
//! ```bash
//! # Run as a standalone server
//! janet-ai-mcp --root /path/to/project
//! ```
//!
//! ### 3. Use as a library
//! ```no_run
//! use janet_ai_mcp::run_server;
//! use std::path::PathBuf;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Start MCP server programmatically
//! run_server(Some(PathBuf::from("/path/to/project"))).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## MCP Tools
//!
//! The server exposes these tools to MCP clients:
//!
//! ### `status`
//! Get comprehensive system status including:
//! - Index statistics and health
//! - Configuration information
//! - Performance metrics
//! - Troubleshooting information
//!
//! ### `regex_search`
//! Search files using regular expressions with:
//! - Pattern matching across file contents
//! - Configurable context lines around matches
//! - File path filtering
//! - Result limiting and pagination
//!
//! ### `semantic_search`
//! Perform AI-powered semantic search with:
//! - Natural language queries
//! - Similarity-based ranking
//! - Embedding vector comparisons
//! - Configurable similarity thresholds
//!
//! ## Configuration
//!
//! The server requires minimal configuration - just point it to a directory
//! containing a `.janet-ai.db` index file created by janet-ai-retriever.
//!
//! ## Integration with Claude Desktop
//!
//! Add this to your Claude Desktop MCP configuration:
//! ```json
//! {
//!   "mcpServers": {
//!     "janet-ai": {
//!       "command": "janet-ai-mcp",
//!       "args": ["--root", "/path/to/your/project"]
//!     }
//!   }
//! }
//! ```

mod server;
pub mod tools;

use server::JanetMcpServer;

use anyhow::Result;
use std::path::PathBuf;
use tracing::info;

/// Configuration for the Janet MCP server.
///
/// This structure holds the configuration parameters needed to initialize
/// and run the Janet AI MCP (Model Context Protocol) server. The server
/// provides semantic and regex search capabilities across codebases.
///
/// # Fields
/// * `root_dir` - The root directory containing the codebase to be searched.
///   This should contain a `.janet-ai.db` file created by the indexing process.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Root directory to search for files
    pub root_dir: PathBuf,
}

impl ServerConfig {
    /// Create a new server configuration with the specified root directory.
    ///
    /// # Arguments
    /// * `root_dir` - Path to the root directory containing the indexed codebase
    ///
    /// # Returns
    /// A new ServerConfig instance
    ///
    /// # Example
    pub fn new(root_dir: PathBuf) -> Self {
        Self { root_dir }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            root_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

/// Run the Janet MCP server with the specified configuration.
///
/// This function initializes and starts the Janet AI MCP server, which provides
/// semantic and regex search capabilities over an indexed codebase. The server
/// communicates using the Model Context Protocol (MCP) over stdio.
///
/// ## Prerequisites
///
/// Before running the server, you must have an indexed database:
/// ```bash
/// # Index your codebase first
/// janet-ai-retriever index --repo .
/// ```
///
/// ## Available Tools
///
/// The server provides these MCP tools to client applications:
/// - **status**: Get comprehensive system status and health information
/// - **regex_search**: Search files using regex patterns with context
/// - **semantic_search**: Perform AI-powered semantic similarity search
///
/// # Arguments
/// * `root_dir` - Optional root directory path. If None, uses current directory
///
/// # Returns
/// `Ok(())` when the server shuts down gracefully
///
/// # Errors
/// - If the `.janet-ai.db` index file is not found in the root directory
/// - Database connection or initialization errors
/// - MCP protocol communication errors
///
/// # Example Usage as Library
///
/// # Example Usage as Binary
/// ```bash
/// # Start MCP server for current directory
/// janet-ai-mcp
///
/// # Start MCP server for specific directory
/// janet-ai-mcp --root /path/to/project
/// ```
pub async fn run_server(root_dir: Option<PathBuf>) -> Result<()> {
    let config = match root_dir {
        Some(dir) => ServerConfig::new(dir),
        None => ServerConfig::default(),
    };
    info!("Starting Janet MCP server");

    // Create the MCP server instance and serve with stdio
    let janet_server = JanetMcpServer::new(config).await?;

    info!("Janet MCP server initialized, starting stdio transport");

    // Serve with stdio transport using rmcp's ServerHandler
    janet_server.serve_stdio().await?;
    Ok(())
}
