use crate::conversation::Conversation;
use gtk4::prelude::*;
use janet_ai_mcp::tools::regex_search::RegexSearchRequest;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub fn create_ui() -> gtk4::Application {
    let app = gtk4::Application::builder()
        .application_id("org.janet-ai.gtk")
        .build();

    let conversation = Arc::new(Mutex::new(Conversation::new(
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    )));

    app.connect_activate(move |app| {
        let window = gtk4::ApplicationWindow::builder()
            .application(app)
            .title("Janet AI Search - MVP")
            .default_width(800)
            .default_height(600)
            .build();

        let main_box = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(12)
            .margin_top(16)
            .margin_bottom(16)
            .margin_start(16)
            .margin_end(16)
            .build();

        let header = gtk4::HeaderBar::builder()
            .title_widget(&gtk4::Label::new(Some("Janet AI Search")))
            .build();
        window.set_titlebar(Some(&header));

        // Search input area
        let search_box = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(8)
            .build();

        let label = gtk4::Label::new(Some("Regex Pattern:"));
        let pattern_entry = gtk4::Entry::builder()
            .hexpand(true)
            .placeholder_text("Enter regex pattern...")
            .build();
        let search_button = gtk4::Button::with_label("Search");

        search_box.append(&label);
        search_box.append(&pattern_entry);
        search_box.append(&search_button);

        // Results area
        let separator = gtk4::Separator::new(gtk4::Orientation::Horizontal);

        let scrolled = gtk4::ScrolledWindow::builder()
            .vexpand(true)
            .hscrollbar_policy(gtk4::PolicyType::Never)
            .vscrollbar_policy(gtk4::PolicyType::Automatic)
            .build();

        let results_view = gtk4::TextView::builder()
            .editable(false)
            .wrap_mode(gtk4::WrapMode::Word)
            .monospace(true)
            .build();

        let initial_text = "Enter a regex pattern above to search the codebase.\n\nMake sure you have indexed your codebase first with:\ncargo run -p janet-ai-retriever -- index --repo .";
        results_view.buffer().set_text(initial_text);

        scrolled.set_child(Some(&results_view));

        main_box.append(&search_box);
        main_box.append(&separator);
        main_box.append(&scrolled);

        window.set_child(Some(&main_box));

        // Connect search functionality
        let results_view_clone = results_view.clone();
        let pattern_entry_clone = pattern_entry.clone();
        let conversation_clone = conversation.clone();

        let perform_search = move || {
            let pattern = pattern_entry_clone.text().to_string();
            if pattern.trim().is_empty() {
                return;
            }

            // Update UI to show searching
            results_view_clone.buffer().set_text(&format!("🔍 Searching for pattern: {pattern}\n⏳ Please wait..."));

            let results_view_inner = results_view_clone.clone();
            let conversation_inner = conversation_clone.clone();
            let pattern_inner = pattern.clone();

            // Spawn async search
            gtk4::glib::spawn_future_local(async move {
                let request = RegexSearchRequest {
                    pattern: pattern_inner,
                    globs: None,
                    include_deps: None,
                    include_docs: None,
                };

                let config = {
                    let conv = conversation_inner.lock().unwrap();
                    conv.server_config.clone()
                };

                let result = janet_ai_mcp::tools::regex_search::regex_search(&config, request).await;

                let display_result = match result {
                    Ok(output) => output,
                    Err(error) => format!("❌ Error: {error}"),
                };

                // Update UI with results
                results_view_inner.buffer().set_text(&display_result);
            });
        };

        // Connect button click
        let perform_search_clone = perform_search.clone();
        search_button.connect_clicked(move |_| {
            perform_search_clone();
        });

        // Connect enter key
        pattern_entry.connect_activate(move |_| {
            perform_search();
        });

        window.present();
    });

    app
}
