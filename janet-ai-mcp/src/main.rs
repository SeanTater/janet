use anyhow::Result;
use clap::{Arg, Command};
use janet_ai_mcp::run_server;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing to stderr (stdout is used for MCP JSON protocol)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    // Parse command line arguments
    let matches = Command::new("janet-ai-mcp")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Janet AI Model Context Protocol server")
        .arg(
            Arg::new("root")
                .short('r')
                .long("root")
                .value_name("DIR")
                .help("Root directory to search")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .get_matches();

    // Get root directory from CLI arguments
    let root_dir = matches.get_one::<PathBuf>("root").cloned();

    // Run the server
    run_server(root_dir).await
}
