mod ui;

use clap::Parser;
use relm4::prelude::*;
use std::path::PathBuf;
use ui::App;

#[derive(Parser)]
#[command(name = "janet-ai-gtk")]
#[command(about = "Janet AI Chat - GTK interface for semantic code search")]
#[command(version)]
struct Args {
    /// Repository path to open (if not provided, shows folder selection dialog)
    repo: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let app = RelmApp::new("org.janet-ai.gtk");
    app.run::<App>(args.repo);
}
