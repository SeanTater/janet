use chrono::{DateTime, Utc};
use janet_ai_mcp::{
    ServerConfig,
    tools::regex_search::{RegexSearchRequest, regex_search},
    tools::semantic_search::{SemanticSearchRequest, semantic_search},
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolType {
    RegexSearch,
    SemanticSearch,
    Status,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolRequest {
    RegexSearch(RegexSearchRequest),
    SemanticSearch(SemanticSearchRequest),
    Status,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool: ToolType,
    pub request: ToolRequest,
    pub response: Option<Result<String, String>>,
    pub timestamp: DateTime<Utc>,
    pub id: String,
}

#[allow(dead_code)]
impl ToolCall {
    pub fn new(tool: ToolType, request: ToolRequest) -> Self {
        Self {
            tool,
            request,
            response: None,
            timestamp: Utc::now(),
            id: uuid::Uuid::new_v4().to_string(),
        }
    }

    pub fn is_pending(&self) -> bool {
        self.response.is_none()
    }

    pub fn is_success(&self) -> bool {
        matches!(self.response, Some(Ok(_)))
    }

    pub fn is_error(&self) -> bool {
        matches!(self.response, Some(Err(_)))
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Conversation {
    pub history: Vec<ToolCall>,
    current_tool: ToolType,
    pub server_config: ServerConfig,
}

#[allow(dead_code)]
impl Conversation {
    pub fn new(root_dir: PathBuf) -> Self {
        Self {
            history: Vec::new(),
            current_tool: ToolType::RegexSearch,
            server_config: ServerConfig::new(root_dir),
        }
    }

    pub fn set_current_tool(&mut self, tool: ToolType) {
        self.current_tool = tool;
    }

    pub fn current_tool(&self) -> &ToolType {
        &self.current_tool
    }

    pub fn history(&self) -> &[ToolCall] {
        &self.history
    }

    pub fn add_call(&mut self, tool: ToolType, request: ToolRequest) -> String {
        let call = ToolCall::new(tool, request);
        let id = call.id.clone();
        self.history.push(call);
        id
    }

    pub async fn execute_call(&mut self, call_id: &str) -> Result<(), String> {
        let call_index = self
            .history
            .iter()
            .position(|call| call.id == call_id)
            .ok_or("Call not found")?;

        let call = &self.history[call_index];
        if call.response.is_some() {
            return Err("Call already executed".to_string());
        }

        let response = match &call.request {
            ToolRequest::RegexSearch(req) => regex_search(&self.server_config, req.clone()).await,
            ToolRequest::SemanticSearch(req) => {
                semantic_search(&self.server_config, req.clone()).await
            }
            ToolRequest::Status => self.get_status().await,
        };

        self.history[call_index].response = Some(response);
        Ok(())
    }

    async fn get_status(&self) -> Result<String, String> {
        use janet_ai_retriever::retrieval::indexing_engine::{
            IndexingEngine, IndexingEngineConfig,
        };

        let indexing_config =
            IndexingEngineConfig::new("janet-gtk".to_string(), self.server_config.root_dir.clone());

        match IndexingEngine::new(indexing_config).await {
            Ok(engine) => {
                let status = engine.get_stats().await;
                Ok(format!("{status:#?}"))
            }
            Err(e) => Err(format!("Failed to get status: {e}")),
        }
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn pending_calls(&self) -> Vec<&ToolCall> {
        self.history
            .iter()
            .filter(|call| call.is_pending())
            .collect()
    }
}
