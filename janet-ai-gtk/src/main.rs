mod ui;

use relm4::prelude::*;
use ui::App;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = RelmApp::new("org.janet-ai.gtk");
    app.run::<App>(());
}
