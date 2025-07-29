use crate::conversation::{Conversation, ToolRequest, ToolType};
use gtk4::prelude::*;
use janet_ai_mcp::tools::regex_search::RegexSearchRequest;
use relm4::prelude::*;
use std::path::PathBuf;

#[derive(Debug)]
pub enum AppMsg {
    ExecuteRegexSearch(String),
    ShowResult(String),
}

pub struct App {
    conversation: Conversation,
    result_text: String,
}

#[relm4::component]
impl SimpleComponent for App {
    type Init = ();
    type Input = AppMsg;
    type Output = ();

    view! {
        main_window = gtk4::ApplicationWindow {
            set_title: Some("Janet AI Search - MVP"),
            set_default_size: (800, 600),

            gtk4::Box {
                set_orientation: gtk4::Orientation::Vertical,
                set_spacing: 12,
                set_margin_all: 16,

                gtk4::HeaderBar {
                    #[wrap(Some)]
                    set_title_widget = &gtk4::Label::new(Some("Janet AI Search")),
                },

                // Simple regex search for MVP
                gtk4::Box {
                    set_orientation: gtk4::Orientation::Horizontal,
                    set_spacing: 8,

                    gtk4::Label {
                        set_text: "Regex Pattern:",
                    },

                    #[name = "pattern_entry"]
                    gtk4::Entry {
                        set_hexpand: true,
                        set_placeholder_text: Some("Enter regex pattern..."),
                        connect_activate[sender] => move |entry| {
                            let pattern = entry.text().to_string();
                            if !pattern.trim().is_empty() {
                                sender.input(AppMsg::ExecuteRegexSearch(pattern));
                            }
                        },
                    },

                    gtk4::Button {
                        set_label: "Search",
                        connect_clicked[sender, pattern_entry] => move |_| {
                            let pattern = pattern_entry.text().to_string();
                            if !pattern.trim().is_empty() {
                                sender.input(AppMsg::ExecuteRegexSearch(pattern));
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
        let model = App {
            conversation: Conversation::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
            result_text: "Enter a regex pattern above to search the codebase.\n\nMake sure you have indexed your codebase first with:\ncargo run -p janet-ai-retriever -- index --repo .".to_string(),
        };

        let widgets = view_output!();

        // Set initial text
        widgets.results_view.buffer().set_text(&model.result_text);

        ComponentParts { model, widgets }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            AppMsg::ExecuteRegexSearch(pattern) => {
                self.result_text =
                    format!("🔍 Searching for pattern: {}\n⏳ Please wait...", pattern);

                let request = ToolRequest::RegexSearch(RegexSearchRequest {
                    pattern: pattern.clone(),
                    globs: None,
                    include_deps: None,
                    include_docs: None,
                });

                let _call_id = self
                    .conversation
                    .add_call(ToolType::RegexSearch, request.clone());

                let sender_clone = sender.clone();
                let config = self.conversation.server_config.clone();
                tokio::spawn(async move {
                    let result = match request {
                        ToolRequest::RegexSearch(req) => {
                            janet_ai_mcp::tools::regex_search::regex_search(&config, req).await
                        }
                        _ => unreachable!(),
                    };

                    let display_result = match result {
                        Ok(output) => output,
                        Err(error) => format!("❌ Error: {}", error),
                    };

                    sender_clone.input(AppMsg::ShowResult(display_result));
                });
            }

            AppMsg::ShowResult(result) => {
                self.result_text = result;
            }
        }
    }
}
