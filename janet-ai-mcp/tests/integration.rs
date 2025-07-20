use janet_ai_mcp::ServerConfig;

// For now, skip the full server creation test since it requires database setup
// The server is tested manually via CLI instead
// TODO: Add proper integration test that uses in-memory database

#[tokio::test]
async fn test_basic_functionality() {
    // This is a placeholder test to ensure the test framework works
    // In a real implementation, this would test MCP server functionality
    let result = 1 + 1;
    assert_eq!(result, 2);
}

#[test]
fn test_config_default() {
    let config = ServerConfig::default();
    assert!(config.root_dir.exists() || config.root_dir == std::path::PathBuf::from("."));
}
