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
    SelectFolder,
    FolderSelected(PathBuf),
}

pub struct App {
    server_config: Option<ServerConfig>,
    current_repo: Option<PathBuf>,
    messages: Vec<ChatMessage>,
    messages_box: gtk4::Box,
    chat_scrolled: gtk4::ScrolledWindow,
    message_entry: gtk4::Entry,
    chat_interface_box: gtk4::Box,
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
    type Init = Option<PathBuf>;
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

            gtk4::Stack {
                // Folder selection screen
                #[name = "folder_selection_box"]
                gtk4::Box {
                    set_orientation: gtk4::Orientation::Vertical,
                    set_spacing: 20,
                    set_margin_all: 40,
                    set_halign: gtk4::Align::Center,
                    set_valign: gtk4::Align::Center,

                    gtk4::Label {
                        set_markup: "<span size='24000' weight='bold'>Welcome to Janet AI</span>",
                        set_margin_bottom: 10,
                    },

                    gtk4::Label {
                        set_text: "Select a repository to start searching your code with semantic AI.",
                        set_margin_bottom: 20,
                    },

                    gtk4::Button {
                        set_label: "📁 Select Repository Folder",
                        set_size_request: (300, 50),
                        add_css_class: "suggested-action",
                        connect_clicked[sender] => move |_| {
                            sender.input(AppMsg::SelectFolder);
                        },
                    },

                    gtk4::Label {
                        set_markup: "<span size='small' color='#666'>Tip: Index your repository first with:\ncargo run -p janet-ai-retriever -- index --repo /path/to/repo</span>",
                        set_margin_top: 20,
                        set_justify: gtk4::Justification::Center,
                    },
                },

                // Chat interface
                #[name = "chat_interface_box"]
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
                            add_css_class: "send-button",
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
            },
        }
    }

    fn init(
        repo_path: Self::Init,
        root: Self::Root,
        sender: ComponentSender<Self>,
    ) -> ComponentParts<Self> {
        let widgets = view_output!();

        // Load CSS for styling
        let css_provider = gtk4::CssProvider::new();
        css_provider.load_from_data(include_str!("style.css"));

        gtk4::style_context_add_provider_for_display(
            &gtk4::gdk::Display::default().unwrap(),
            &css_provider,
            gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );

        let (server_config, current_repo, messages, show_chat) = if let Some(repo) = repo_path {
            // Repository provided via command line
            let config = ServerConfig::new(repo.clone());
            let messages = vec![ChatMessage::new_system(format!(
                "Welcome to Janet AI Chat! 🤖\n\nRepository: {}\n\nUse commands like:\n• /regex pattern - Search with regex\n• /semantic query - Semantic search\n• query - Default semantic search\n\nMake sure you've indexed your codebase first:\ncargo run -p janet-ai-retriever -- index --repo .",
                repo.display()
            ))];
            (Some(config), Some(repo), messages, true)
        } else {
            // No repository provided, show folder selection
            (None, None, Vec::new(), false)
        };

        let model = App {
            server_config,
            current_repo,
            messages,
            messages_box: widgets.messages_box.clone(),
            chat_scrolled: widgets.chat_scrolled.clone(),
            message_entry: widgets.message_entry.clone(),
            chat_interface_box: widgets.chat_interface_box.clone(),
        };

        // Show appropriate interface - get the Stack from the main window
        let stack = root.child().unwrap().downcast::<gtk4::Stack>().unwrap();
        if show_chat {
            stack.set_visible_child(&widgets.chat_interface_box);
            // Add initial welcome message to UI
            if !model.messages.is_empty() {
                let message_widget = Self::create_message_bubble(&model.messages[0]);
                model.messages_box.append(&message_widget);
            }
        } else {
            stack.set_visible_child(&widgets.folder_selection_box);
        }

        ComponentParts { model, widgets }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            AppMsg::SendMessage(input) => {
                if let Some(config) = &self.server_config {
                    let (search_type, query) = Self::parse_command(&input);
                    let user_message = ChatMessage::new_user(input, search_type.clone());
                    self.messages.push(user_message.clone());

                    Self::add_message_and_scroll(
                        &self.messages_box,
                        &self.chat_scrolled,
                        &user_message,
                    );

                    let config = config.clone();
                    relm4::spawn(async move {
                        let result = Self::execute_search(&config, search_type, query).await;
                        let display_result =
                            result.unwrap_or_else(|error| format!("❌ Error: {error}"));
                        sender.input(AppMsg::ShowResult(display_result));
                    });
                }
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

            AppMsg::SelectFolder => {
                let dialog = gtk4::FileChooserDialog::new(
                    Some("Select Repository Folder"),
                    None::<&gtk4::Window>,
                    gtk4::FileChooserAction::SelectFolder,
                    &[
                        ("Cancel", gtk4::ResponseType::Cancel),
                        ("Select", gtk4::ResponseType::Accept),
                    ],
                );

                let sender_clone = sender.clone();
                dialog.connect_response(move |dialog, response| {
                    if response == gtk4::ResponseType::Accept {
                        if let Some(file) = dialog.file() {
                            if let Some(path) = file.path() {
                                sender_clone.input(AppMsg::FolderSelected(path));
                            }
                        }
                    }
                    dialog.close();
                });

                dialog.show();
            }

            AppMsg::FolderSelected(repo_path) => {
                // Set up the repository
                self.server_config = Some(ServerConfig::new(repo_path.clone()));
                self.current_repo = Some(repo_path.clone());

                // Initialize with welcome message
                let welcome_message = ChatMessage::new_system(format!(
                    "Welcome to Janet AI Chat! 🤖\n\nRepository: {}\n\nUse commands like:\n• /regex pattern - Search with regex\n• /semantic query - Semantic search\n• query - Default semantic search\n\nMake sure you've indexed your codebase first:\ncargo run -p janet-ai-retriever -- index --repo .",
                    repo_path.display()
                ));
                self.messages = vec![welcome_message.clone()];

                // Switch to chat interface - get the Stack from the main window
                let main_window = self
                    .chat_interface_box
                    .root()
                    .unwrap()
                    .downcast::<gtk4::ApplicationWindow>()
                    .unwrap();
                let stack = main_window
                    .child()
                    .unwrap()
                    .downcast::<gtk4::Stack>()
                    .unwrap();
                stack.set_visible_child(&self.chat_interface_box);

                // Clear existing messages and add welcome message
                while let Some(child) = self.messages_box.first_child() {
                    self.messages_box.remove(&child);
                }
                let message_widget = Self::create_message_bubble(&welcome_message);
                self.messages_box.append(&message_widget);
            }
        }
    }
}
