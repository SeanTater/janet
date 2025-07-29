use gtk4::prelude::*;
use janet_ai_mcp::{
    ServerConfig,
    tools::{regex_search::RegexSearchRequest, semantic_search::SemanticSearchRequest},
};
use relm4::prelude::*;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum SearchType {
    Regex,
    Semantic,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub content: String,
    pub is_user: bool,
    pub search_type: Option<SearchType>,
    pub timestamp: String,
}

impl ChatMessage {
    pub fn new_user(content: String, search_type: SearchType) -> Self {
        Self {
            content,
            is_user: true,
            search_type: Some(search_type),
            timestamp: chrono::Local::now().format("%H:%M").to_string(),
        }
    }

    pub fn new_system(content: String) -> Self {
        Self {
            content,
            is_user: false,
            search_type: None,
            timestamp: chrono::Local::now().format("%H:%M").to_string(),
        }
    }
}

#[derive(Debug)]
pub enum AppMsg {
    SendMessage(String),
    ShowResult(String),
    ClearEntry,
}

pub struct App {
    server_config: ServerConfig,
    messages: Vec<ChatMessage>,
    messages_box: gtk4::Box,
    chat_scrolled: gtk4::ScrolledWindow,
    message_entry: gtk4::Entry,
}

impl App {
    /// Parse IRC-style commands from user input
    fn parse_command(input: &str) -> (SearchType, String) {
        let input = input.trim();

        if let Some(rest) = input.strip_prefix("/regex ") {
            (SearchType::Regex, rest.to_string())
        } else if let Some(rest) = input.strip_prefix("/semantic ") {
            (SearchType::Semantic, rest.to_string())
        } else {
            // Default to semantic search for plain queries
            (SearchType::Semantic, input.to_string())
        }
    }

    /// Create a message bubble widget and add it to the container
    fn add_message_to_ui(container: &gtk4::Box, message: &ChatMessage) {
        let message_widget = Self::create_message_bubble(message);
        container.append(&message_widget);
    }

    /// Create a styled message bubble
    fn create_message_bubble(message: &ChatMessage) -> gtk4::Widget {
        let message_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);

        if message.is_user {
            // User message - aligned right
            message_box.set_halign(gtk4::Align::End);

            let bubble = gtk4::Frame::new(None);
            bubble.set_css_classes(&["user-message"]);

            let content_box = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
            content_box.set_margin_all(12);

            // Message content
            let text_label = gtk4::Label::new(Some(&message.content));
            text_label.set_wrap(true);
            text_label.set_xalign(0.0);
            text_label.set_selectable(true);

            // Timestamp and search type
            let meta_text = if let Some(ref search_type) = message.search_type {
                format!(
                    "{} • {}",
                    message.timestamp,
                    match search_type {
                        SearchType::Regex => "regex",
                        SearchType::Semantic => "semantic",
                    }
                )
            } else {
                message.timestamp.clone()
            };

            let meta_label = gtk4::Label::new(Some(&meta_text));
            meta_label.set_css_classes(&["message-meta"]);
            meta_label.set_xalign(1.0);

            content_box.append(&text_label);
            content_box.append(&meta_label);
            bubble.set_child(Some(&content_box));

            message_box.append(&bubble);
        } else {
            // System message - aligned left
            message_box.set_halign(gtk4::Align::Start);

            let bubble = gtk4::Frame::new(None);
            bubble.set_css_classes(&["system-message"]);

            let content_box = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
            content_box.set_margin_all(12);

            // Message content
            let text_label = gtk4::Label::new(Some(&message.content));
            text_label.set_wrap(true);
            text_label.set_xalign(0.0);
            text_label.set_selectable(true);
            text_label.set_css_classes(&["monospace"]);

            // Timestamp
            let meta_label = gtk4::Label::new(Some(&message.timestamp));
            meta_label.set_css_classes(&["message-meta"]);
            meta_label.set_xalign(0.0);

            content_box.append(&text_label);
            content_box.append(&meta_label);
            bubble.set_child(Some(&content_box));

            message_box.append(&bubble);
        }

        message_box.upcast()
    }

    /// Scroll the chat to the bottom
    fn scroll_to_bottom(scrolled_window: &gtk4::ScrolledWindow) {
        // Use a timeout to ensure the widget is updated
        gtk4::glib::timeout_add_local_once(std::time::Duration::from_millis(10), {
            let scrolled = scrolled_window.clone();
            move || {
                let vadj = scrolled.vadjustment();
                vadj.set_value(vadj.upper() - vadj.page_size());
            }
        });
    }
}

#[relm4::component(pub)]
impl SimpleComponent for App {
    type Init = ();
    type Input = AppMsg;
    type Output = ();

    view! {
        main_window = gtk4::ApplicationWindow {
            set_default_size: (800, 600),

            #[wrap(Some)]
            set_titlebar = &gtk4::HeaderBar {
                #[wrap(Some)]
                set_title_widget = &gtk4::Label::new(Some("Janet AI Chat")),
            },

            gtk4::Box {
                set_orientation: gtk4::Orientation::Vertical,
                set_spacing: 0,

                // Chat messages area
                #[name = "chat_scrolled"]
                gtk4::ScrolledWindow {
                    set_vexpand: true,
                    set_hscrollbar_policy: gtk4::PolicyType::Never,
                    set_vscrollbar_policy: gtk4::PolicyType::Automatic,
                    set_margin_all: 8,

                    #[name = "messages_box"]
                    gtk4::Box {
                        set_orientation: gtk4::Orientation::Vertical,
                        set_spacing: 8,
                        set_margin_all: 8,
                        set_valign: gtk4::Align::End,
                    },
                },

                // Input area at bottom
                gtk4::Box {
                    set_orientation: gtk4::Orientation::Horizontal,
                    set_spacing: 8,
                    set_margin_all: 8,

                    #[name = "message_entry"]
                    gtk4::Entry {
                        set_hexpand: true,
                        set_placeholder_text: Some("Type '/regex pattern' or '/semantic query' or just 'query' for semantic search..."),
                        connect_activate[sender] => move |entry| {
                            let text = entry.text().to_string();
                            if !text.trim().is_empty() {
                                sender.input(AppMsg::SendMessage(text));
                                sender.input(AppMsg::ClearEntry);
                            }
                        },
                    },

                    gtk4::Button {
                        set_label: "Send",
                        connect_clicked[sender, message_entry] => move |_| {
                            let text = message_entry.text().to_string();
                            if !text.trim().is_empty() {
                                sender.input(AppMsg::SendMessage(text));
                                sender.input(AppMsg::ClearEntry);
                            }
                        },
                    },
                },
            },
        }
    }

    fn init(
        _init: Self::Init,
        root: Self::Root,
        sender: ComponentSender<Self>,
    ) -> ComponentParts<Self> {
        let messages = vec![ChatMessage::new_system(
            "Welcome to Janet AI Chat! 🤖\n\nUse commands like:\n• /regex pattern - Search with regex\n• /semantic query - Semantic search\n• query - Default semantic search\n\nMake sure you've indexed your codebase first:\ncargo run -p janet-ai-retriever -- index --repo .".to_string()
        )];

        let widgets = view_output!();

        // Load CSS for styling
        let css_provider = gtk4::CssProvider::new();
        css_provider.load_from_data(include_str!("style.css"));

        gtk4::style_context_add_provider_for_display(
            &gtk4::gdk::Display::default().unwrap(),
            &css_provider,
            gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );

        let model = App {
            server_config: ServerConfig::new(
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            ),
            messages,
            messages_box: widgets.messages_box.clone(),
            chat_scrolled: widgets.chat_scrolled.clone(),
            message_entry: widgets.message_entry.clone(),
        };

        // Add initial welcome message to UI
        Self::add_message_to_ui(&model.messages_box, &model.messages[0]);

        ComponentParts { model, widgets }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            AppMsg::SendMessage(input) => {
                let (search_type, query) = Self::parse_command(&input);

                // Add user message
                let user_message = ChatMessage::new_user(input, search_type.clone());
                self.messages.push(user_message.clone());

                // Add message to UI
                Self::add_message_to_ui(&self.messages_box, &user_message);

                // Scroll to bottom
                Self::scroll_to_bottom(&self.chat_scrolled);

                // Execute search
                let sender_clone = sender.clone();
                let config = self.server_config.clone();

                relm4::spawn(async move {
                    let result = match search_type {
                        SearchType::Regex => {
                            let req = RegexSearchRequest {
                                pattern: query,
                                globs: None,
                                include_deps: None,
                                include_docs: None,
                            };
                            janet_ai_mcp::tools::regex_search::regex_search(&config, req).await
                        }
                        SearchType::Semantic => {
                            let req = SemanticSearchRequest {
                                query,
                                limit: Some(20),
                                threshold: None,
                            };
                            janet_ai_mcp::tools::semantic_search::semantic_search(&config, req)
                                .await
                        }
                    };

                    let display_result = match result {
                        Ok(output) => output,
                        Err(error) => format!("❌ Error: {error}"),
                    };

                    sender_clone.input(AppMsg::ShowResult(display_result));
                });
            }

            AppMsg::ShowResult(result) => {
                // Add system response message
                let response_message = ChatMessage::new_system(result);
                self.messages.push(response_message.clone());

                // Add message to UI
                Self::add_message_to_ui(&self.messages_box, &response_message);

                // Scroll to bottom
                Self::scroll_to_bottom(&self.chat_scrolled);
            }

            AppMsg::ClearEntry => {
                self.message_entry.set_text("");
            }
        }
    }
}
