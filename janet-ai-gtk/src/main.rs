mod conversation;
mod simple_ui;

use gtk4::prelude::*;
use simple_ui::create_ui;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = create_ui();
    app.run();
}
