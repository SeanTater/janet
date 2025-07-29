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

#[derive(Debug)]
pub enum AppMsg {
    ExecuteSearch(String),
    ShowResult(String),
    SetSearchType(SearchType),
}

pub struct App {
    server_config: ServerConfig,
    result_text: String,
    search_type: SearchType,
    text_buffer: gtk4::TextBuffer,
}

#[relm4::component(pub)]
impl SimpleComponent for App {
    type Init = ();
    type Input = AppMsg;
    type Output = ();

    view! {
        main_window = gtk4::ApplicationWindow {
            set_default_size: (900, 700),

            #[wrap(Some)]
            set_titlebar = &gtk4::HeaderBar {
                #[wrap(Some)]
                set_title_widget = &gtk4::Label::new(Some("Janet AI Search")),
            },

            gtk4::Box {
                set_orientation: gtk4::Orientation::Vertical,
                set_spacing: 12,
                set_margin_all: 16,

                // Search type selection
                gtk4::Box {
                    set_orientation: gtk4::Orientation::Horizontal,
                    set_spacing: 12,

                    gtk4::Label {
                        set_text: "Search Type:",
                    },

                    #[name = "regex_radio"]
                    gtk4::CheckButton {
                        set_label: Some("Regex"),
                        set_active: true,
                        connect_toggled[sender] => move |btn| {
                            if btn.is_active() {
                                sender.input(AppMsg::SetSearchType(SearchType::Regex));
                            }
                        },
                    },

                    #[name = "semantic_radio"]
                    gtk4::CheckButton {
                        set_label: Some("Semantic"),
                        set_group: Some(&regex_radio),
                        connect_toggled[sender] => move |btn| {
                            if btn.is_active() {
                                sender.input(AppMsg::SetSearchType(SearchType::Semantic));
                            }
                        },
                    },
                },

                // Search input area
                gtk4::Box {
                    set_orientation: gtk4::Orientation::Horizontal,
                    set_spacing: 8,

                    #[name = "search_label"]
                    gtk4::Label {
                        #[watch]
                        set_text: match model.search_type {
                            SearchType::Regex => "Regex Pattern:",
                            SearchType::Semantic => "Semantic Query:",
                        },
                    },

                    #[name = "search_entry"]
                    gtk4::Entry {
                        set_hexpand: true,
                        #[watch]
                        set_placeholder_text: match model.search_type {
                            SearchType::Regex => Some("Enter regex pattern..."),
                            SearchType::Semantic => Some("Enter semantic search query..."),
                        },
                        connect_activate[sender] => move |entry| {
                            let pattern = entry.text().to_string();
                            if !pattern.trim().is_empty() {
                                sender.input(AppMsg::ExecuteSearch(pattern));
                            }
                        },
                    },

                    gtk4::Button {
                        set_label: "Search",
                        connect_clicked[sender, search_entry] => move |_| {
                            let pattern = search_entry.text().to_string();
                            if !pattern.trim().is_empty() {
                                sender.input(AppMsg::ExecuteSearch(pattern));
                            }
                        },
                    },
                },

                gtk4::Separator {},

                // Results area
                gtk4::ScrolledWindow {
                    set_vexpand: true,
                    set_hscrollbar_policy: gtk4::PolicyType::Never,
                    set_vscrollbar_policy: gtk4::PolicyType::Automatic,

                    #[name = "results_view"]
                    gtk4::TextView {
                        set_editable: false,
                        set_wrap_mode: gtk4::WrapMode::Word,
                        set_monospace: true,
                        set_buffer: Some(&model.text_buffer),
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
        let text_buffer = gtk4::TextBuffer::new(None);
        let initial_text = "Select a search type and enter a pattern to search the codebase.\n\n• Regex: Search using regular expressions\n• Semantic: Search using AI-powered semantic similarity\n\nMake sure you have indexed your codebase first with:\ncargo run -p janet-ai-retriever -- index --repo .";
        text_buffer.set_text(initial_text);

        let model = App {
            server_config: ServerConfig::new(
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            ),
            result_text: initial_text.to_string(),
            search_type: SearchType::Regex,
            text_buffer,
        };

        let widgets = view_output!();

        ComponentParts { model, widgets }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            AppMsg::ExecuteSearch(pattern) => {
                let search_desc = match self.search_type {
                    SearchType::Regex => "regex pattern",
                    SearchType::Semantic => "semantic query",
                };
                self.result_text =
                    format!("🔍 Searching for {search_desc}: {pattern}\n⏳ Please wait...");
                self.text_buffer.set_text(&self.result_text);

                let sender_clone = sender.clone();
                let config = self.server_config.clone();
                let search_type = self.search_type.clone();

                relm4::spawn(async move {
                    let result = match search_type {
                        SearchType::Regex => {
                            let req = RegexSearchRequest {
                                pattern,
                                globs: None,
                                include_deps: None,
                                include_docs: None,
                            };
                            janet_ai_mcp::tools::regex_search::regex_search(&config, req).await
                        }
                        SearchType::Semantic => {
                            let req = SemanticSearchRequest {
                                query: pattern,
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

            AppMsg::SetSearchType(search_type) => {
                self.search_type = search_type;
            }

            AppMsg::ShowResult(result) => {
                self.result_text = result;
                self.text_buffer.set_text(&self.result_text);
            }
        }
    }
}
