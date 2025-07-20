use anyhow::Result;
use clap::{Arg, Command};
use janet_ai_mcp::{ServerConfig, run_server};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let matches = Command::new("janet-ai-mcp")
        .version("0.1.0")
        .about("Janet AI Model Context Protocol server")
        .arg(
            Arg::new("root")
                .short('r')
                .long("root")
                .value_name("DIR")
                .help("Root directory to search")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("no-semantic")
                .long("no-semantic")
                .help("Disable semantic search functionality")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("enable-delegate")
                .long("enable-delegate")
                .help("Enable delegate search with LLM validation")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // Build configuration from CLI arguments
    let mut config = ServerConfig::default();

    if let Some(root_dir) = matches.get_one::<PathBuf>("root") {
        config.root_dir = root_dir.clone();
    }

    config.enable_semantic_search = !matches.get_flag("no-semantic");
    config.enable_delegate_search = matches.get_flag("enable-delegate");

    // Run the server
    run_server(config).await
}
