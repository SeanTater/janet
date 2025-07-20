//! # janet-ai-mcp
//!
//! A Model Context Protocol (MCP) server for janet-ai that provides semantic and regex
//! search capabilities across codebases. Integrates with janet-ai-retriever for file
//! indexing and janet-ai-embed for semantic embeddings.

pub mod server;
pub mod tools;

pub use server::JanetMcpServer;

use anyhow::Result;
use std::path::PathBuf;
use tracing::info;

/// Configuration for the Janet MCP server
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Root directory to search for files
    pub root_dir: PathBuf,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            root_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

/// Run the Janet MCP server with the given configuration
pub async fn run_server(config: ServerConfig) -> Result<()> {
    info!("Starting Janet MCP server");

    // Create the MCP server instance and serve with stdio
    let janet_server = JanetMcpServer::new(config).await?;

    info!("Janet MCP server initialized, starting stdio transport");

    // Serve with stdio transport using rmcp's ServerHandler
    janet_server.serve_stdio().await?;

    Ok(())
}
