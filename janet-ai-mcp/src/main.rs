use anyhow::Result;
use clap::Parser;
use janet_ai_mcp::run_server;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Janet AI Model Context Protocol server")]
struct Args {
    #[arg(short, long, value_name = "DIR", help = "Root directory to search")]
    root: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing to stderr (stdout is used for MCP JSON protocol)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    run_server(args.root).await
}
