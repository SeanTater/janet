use janet_ai_mcp::{JanetMcpServer, ServerConfig};
use std::path::PathBuf;

#[tokio::test]
async fn test_server_creation() {
    let config = ServerConfig {
        root_dir: PathBuf::from("."),
        enable_semantic_search: true,
        enable_delegate_search: false,
    };

    let server = JanetMcpServer::new(config).await;
    assert!(server.is_ok());
}

#[test]
fn test_config_default() {
    let config = ServerConfig::default();
    assert!(config.enable_semantic_search);
    assert!(!config.enable_delegate_search);
}
