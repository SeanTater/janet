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
        match input {
            s if s.starts_with("/regex ") => (SearchType::Regex, s[7..].to_string()),
            s if s.starts_with("/semantic ") => (SearchType::Semantic, s[10..].to_string()),
            _ => (SearchType::Semantic, input.to_string()),
        }
    }

    /// Add message to UI and scroll to bottom
    fn add_message_and_scroll(
        container: &gtk4::Box,
        scrolled: &gtk4::ScrolledWindow,
        message: &ChatMessage,
    ) {
        let message_widget = Self::create_message_bubble(message);
        container.append(&message_widget);
        Self::scroll_to_bottom(scrolled);
    }

    /// Create a styled message bubble
    fn create_message_bubble(message: &ChatMessage) -> gtk4::Widget {
        let message_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        let (css_class, align, text_align, meta_align) = if message.is_user {
            ("user-message", gtk4::Align::End, 0.0, 1.0)
        } else {
            ("system-message", gtk4::Align::Start, 0.0, 0.0)
        };

        message_box.set_halign(align);

        let bubble = gtk4::Frame::new(None);
        bubble.set_css_classes(&[css_class]);

        let content_box = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        content_box.set_margin_all(12);

        let text_label = gtk4::Label::new(Some(&message.content));
        text_label.set_wrap(true);
        text_label.set_xalign(text_align);
        text_label.set_selectable(true);
        if !message.is_user {
            text_label.set_css_classes(&["monospace"]);
        }

        let meta_text = if message.is_user {
            message
                .search_type
                .as_ref()
                .map(|st| {
                    format!(
                        "{} • {}",
                        message.timestamp,
                        match st {
                            SearchType::Regex => "regex",
                            SearchType::Semantic => "semantic",
                        }
                    )
                })
                .unwrap_or_else(|| message.timestamp.clone())
        } else {
            message.timestamp.clone()
        };

        let meta_label = gtk4::Label::new(Some(&meta_text));
        meta_label.set_css_classes(&["message-meta"]);
        meta_label.set_xalign(meta_align);

        content_box.append(&text_label);
        content_box.append(&meta_label);
        bubble.set_child(Some(&content_box));
        message_box.append(&bubble);

        message_box.upcast()
    }

    /// Execute search request based on type
    async fn execute_search(
        config: &ServerConfig,
        search_type: SearchType,
        query: String,
    ) -> Result<String, String> {
        match search_type {
            SearchType::Regex => {
                let req = RegexSearchRequest {
                    pattern: query,
                    globs: None,
                    include_deps: None,
                    include_docs: None,
                };
                janet_ai_mcp::tools::regex_search::regex_search(config, req).await
            }
            SearchType::Semantic => {
                let req = SemanticSearchRequest {
                    query,
                    limit: Some(20),
                    threshold: None,
                };
                janet_ai_mcp::tools::semantic_search::semantic_search(config, req).await
            }
        }
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
        let message_widget = Self::create_message_bubble(&model.messages[0]);
        model.messages_box.append(&message_widget);

        ComponentParts { model, widgets }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            AppMsg::SendMessage(input) => {
                let (search_type, query) = Self::parse_command(&input);
                let user_message = ChatMessage::new_user(input, search_type.clone());
                self.messages.push(user_message.clone());

                Self::add_message_and_scroll(
                    &self.messages_box,
                    &self.chat_scrolled,
                    &user_message,
                );

                let config = self.server_config.clone();
                relm4::spawn(async move {
                    let result = Self::execute_search(&config, search_type, query).await;
                    let display_result =
                        result.unwrap_or_else(|error| format!("❌ Error: {error}"));
                    sender.input(AppMsg::ShowResult(display_result));
                });
            }

            AppMsg::ShowResult(result) => {
                let response_message = ChatMessage::new_system(result);
                self.messages.push(response_message.clone());
                Self::add_message_and_scroll(
                    &self.messages_box,
                    &self.chat_scrolled,
                    &response_message,
                );
            }

            AppMsg::ClearEntry => {
                self.message_entry.set_text("");
            }
        }
    }
}
