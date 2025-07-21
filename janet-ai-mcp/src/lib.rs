//! # janet-ai-mcp
//!
//! A Model Context Protocol (MCP) server for janet-ai that provides semantic and regex
//! search capabilities across codebases. Integrates with janet-ai-retriever for file
//! indexing and janet-ai-embed for semantic embeddings.

mod server;
mod tools;

use server::JanetMcpServer;

use anyhow::Result;
use std::path::PathBuf;
use tracing::info;

/// Configuration for the Janet MCP server
#[derive(Debug, Clone)]
struct ServerConfig {
    /// Root directory to search for files
    root_dir: PathBuf,
}

impl ServerConfig {
    fn new(root_dir: PathBuf) -> Self {
        Self { root_dir }
    }

    fn default() -> Self {
        Self {
            root_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

/// Run the Janet MCP server with the given root directory
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
