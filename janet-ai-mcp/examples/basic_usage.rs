use anyhow::Result;
use janet_ai_mcp::{ServerConfig, run_server};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Configure the server
    let config = ServerConfig {
        root_dir: PathBuf::from("."),
        enable_semantic_search: true,
        enable_delegate_search: false,
    };

    // Run the MCP server
    run_server(config).await
}
