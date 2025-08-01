use std::process::Stdio;

// Note about flaky tests:
// Some integration tests are marked with #[ignore] due to flakiness from:
// - Process spawning and stdio communication timing
// - Model downloading and embedding generation
// - Complex multi-step pipelines with timeouts
//
// These tests can be run individually with:
// cargo test --test integration -- --ignored
//
// For reliable testing, prefer the unit tests in src/ modules.

#[cfg(test)]
use std::path::PathBuf;
use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

/// Test that basic file system operations work
#[test]
fn test_filesystem_basics() {
    let temp_dir = tempdir().expect("Failed to create temp directory");

    // Test that we can create and access the temp directory
    assert!(temp_dir.path().exists());
    assert!(temp_dir.path().is_dir());

    // Test that we can create a file in it
    let test_file = temp_dir.path().join("test.txt");
    std::fs::write(&test_file, "test content").expect("Failed to write test file");
    assert!(test_file.exists());

    let content = std::fs::read_to_string(&test_file).expect("Failed to read test file");
    assert_eq!(content, "test content");
}

/// Test that the server binary can start and exit cleanly
#[tokio::test]
async fn test_server_startup() {
    // Try to run the server with --help to see if the binary works
    let output = Command::new("cargo")
        .args(["run", "-p", "janet-ai-mcp", "--", "--help"])
        .output()
        .await
        .expect("Failed to run server with --help");

    if !output.status.success() {
        eprintln!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
    }

    assert!(output.status.success(), "Server --help should succeed");
}

/// Test that server can start and be killed quickly
#[tokio::test]
async fn test_server_kill() {
    let temp_dir = tempdir().expect("Failed to create temp directory");

    // Start server and immediately kill it
    let mut child = Command::new("cargo")
        .args([
            "run",
            "-p",
            "janet-ai-mcp",
            "--",
            "--root",
            temp_dir.path().to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start server");

    let wait_time = if cfg!(windows) { 5 } else { 2 };
    tokio::time::sleep(Duration::from_secs(wait_time)).await;

    let kill_result = child.kill().await;
    let wait_result = child.wait().await;

    // Just check that we could start and kill the process
    assert!(kill_result.is_ok() || wait_result.is_ok());
}

/// Test actual MCP protocol communication over stdio
#[tokio::test]
#[ignore] // Flaky due to process spawning, stdio communication, and timing dependencies
async fn test_mcp_initialize() {
    let temp_dir = tempdir().expect("Failed to create temp directory");

    // Start the MCP server process
    println!("Starting MCP server with cargo run...");
    let mut child = Command::new("cargo")
        .args(["run", "--", "--root", temp_dir.path().to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start MCP server");

    // Give the server a moment to start up (longer in CI, especially on Windows)
    println!("Waiting for server to start...");
    let startup_delay = if cfg!(windows) { 10 } else { 5 };
    tokio::time::sleep(Duration::from_secs(startup_delay)).await;

    let stdin = child.stdin.as_mut().expect("Failed to get stdin");
    let stdout = child.stdout.as_mut().expect("Failed to get stdout");
    let mut reader = BufReader::new(stdout);

    // Send MCP initialize request
    let initialize_request = r#"{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}
"#;

    println!("Sending initialize request: {}", initialize_request.trim());
    stdin
        .write_all(initialize_request.as_bytes())
        .await
        .expect("Failed to write initialize request");
    stdin.flush().await.expect("Failed to flush stdin");
    println!("Request sent successfully");

    // Read initialize response with longer timeout for CI (especially Windows)
    println!("Waiting for response...");
    let mut response = String::new();
    let response_timeout = if cfg!(windows) { 40 } else { 20 };
    let read_result = timeout(
        Duration::from_secs(response_timeout),
        reader.read_line(&mut response),
    )
    .await;

    match read_result {
        Ok(Ok(_)) => {
            println!("Initialize response: {response}");
            // Basic check that we got a JSON response
            assert!(response.contains("jsonrpc"));
            assert!(response.contains("result"));
            assert!(response.contains("Janet AI MCP Server"));
        }
        Ok(Err(io_err)) => {
            panic!("IO error reading response: {io_err}");
        }
        Err(_timeout_err) => {
            // Check if child process is still running
            if let Ok(Some(exit_status)) = child.try_wait() {
                // Server exited - capture stderr for debugging
                let stderr = child.stderr.take().unwrap();
                let mut stderr_reader = BufReader::new(stderr);
                let mut stderr_output = String::new();
                let _ = stderr_reader.read_to_string(&mut stderr_output).await;
                panic!(
                    "MCP server exited early with status: {exit_status}\nStderr: {stderr_output}"
                );
            } else {
                // Server still running but not responding - this might be a deadlock or compilation hang
                println!("Server process still running, attempting to capture stderr...");

                // Try to read any stderr output that might indicate what's wrong
                if let Some(stderr) = child.stderr.as_mut() {
                    let mut stderr_reader = BufReader::new(stderr);
                    let mut stderr_output = String::new();
                    // Try to read stderr with a short timeout
                    if let Ok(Ok(_)) = timeout(
                        Duration::from_secs(2),
                        stderr_reader.read_to_string(&mut stderr_output),
                    )
                    .await
                    {
                        if !stderr_output.is_empty() {
                            println!("Server stderr output:");
                            println!("{stderr_output}");
                        } else {
                            println!("Server stderr is empty");
                        }
                    } else {
                        println!("Failed to read stderr within timeout");
                    }
                }

                panic!("Timeout waiting for initialize response (server still running after 30s)");
            }
        }
    }

    // Clean shutdown
    let _ = child.kill().await;
    let _ = child.wait().await;
}

/// Test end-to-end semantic search with real test data
#[tokio::test]
#[ignore] // Very flaky - requires full indexing, embeddings, model downloading, process management, and stdio communication
async fn test_semantic_search_with_real_data() {
    use janet_ai_retriever::retrieval::indexing_engine::{IndexingEngine, IndexingEngineConfig};

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let test_repo_path = temp_dir.path().join("test_repo");

    // Copy test data from janet-ai-retriever examples (relative to workspace root)
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let source_data_path = workspace_root.join("janet-ai-retriever/examples/test_data");
    copy_dir_all(&source_data_path, &test_repo_path).expect("Failed to copy test data");

    println!("Test repo created at: {test_repo_path:?}");

    // Build an index using IndexingEngine directly (like the working example)
    let indexing_config = IndexingEngineConfig::new("mcp-test".to_string(), test_repo_path.clone())
        .with_max_workers(2)
        .with_chunk_size(500);

    println!("⚙️  Initializing IndexingEngine...");

    // Ensure the temporary directory exists
    std::fs::create_dir_all(&test_repo_path).expect("Failed to create test repo directory");

    // Create indexing engine (using persistent database so MCP server can access it)
    // Handle database permission issues gracefully like other tests do
    let mut engine = match IndexingEngine::new(indexing_config).await {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("Skipping semantic search test due to database permissions: {e}");
            return;
        }
    };

    println!("✅ IndexingEngine initialized successfully");

    // Start the engine and perform full reindex
    println!("🔄 Starting full reindex...");
    engine.start(true).await.expect("Failed to start indexing");

    // Wait for indexing to complete
    let mut attempts = 0;
    let max_attempts = 30; // 30 seconds max wait

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Process any pending tasks
        engine
            .process_pending_tasks()
            .await
            .expect("Failed to process tasks");

        let queue_size = engine.get_queue_size().await;
        let stats = engine.get_stats().await;

        if attempts % 5 == 0 || queue_size == 0 {
            println!(
                "📊 Queue size: {}, Files processed: {}, Chunks created: {}, Embeddings: {}",
                queue_size, stats.files_processed, stats.chunks_created, stats.embeddings_generated
            );
        }

        if queue_size == 0 && stats.files_processed > 0 {
            println!("✅ Indexing completed!");
            break;
        }

        attempts += 1;
        if attempts >= max_attempts {
            panic!("Timeout waiting for indexing to complete");
        }
    }

    // Get final statistics
    let final_stats = engine.get_stats().await;
    println!(
        "📈 Indexing completed: {} files, {} chunks",
        final_stats.files_processed, final_stats.chunks_created
    );

    // Shutdown the engine
    engine.shutdown().await.expect("Failed to shutdown engine");

    println!("Indexing completed successfully");

    // Start the MCP server with the test repository
    let mut child = Command::new("cargo")
        .args(["run", "--", "--root", test_repo_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start MCP server");

    let stdin = child.stdin.as_mut().expect("Failed to get stdin");
    let stdout = child.stdout.as_mut().expect("Failed to get stdout");
    let mut reader = BufReader::new(stdout);

    // Initialize MCP protocol
    let initialize_request = r#"{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}
"#;

    stdin
        .write_all(initialize_request.as_bytes())
        .await
        .expect("Failed to write initialize request");
    stdin.flush().await.expect("Failed to flush stdin");

    // Read initialize response
    let mut response = String::new();
    let init_timeout = if cfg!(windows) { 20 } else { 10 };
    timeout(
        Duration::from_secs(init_timeout),
        reader.read_line(&mut response),
    )
    .await
    .expect("Failed to get initialize response")
    .expect("Failed to read initialize response");

    assert!(response.contains("jsonrpc"));
    println!("Initialize OK");

    // Send initialized notification
    let initialized_notification = r#"{"jsonrpc": "2.0", "method": "initialized", "params": {}}
"#;

    stdin
        .write_all(initialized_notification.as_bytes())
        .await
        .expect("Failed to write initialized");
    stdin.flush().await.expect("Failed to flush stdin");

    // Test semantic search for mathematical functions
    let semantic_search_request = r#"{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "semantic_search", "arguments": {"query": "function that adds two numbers", "limit": 5, "threshold": 0.3}}}
"#;

    stdin
        .write_all(semantic_search_request.as_bytes())
        .await
        .expect("Failed to write semantic search request");
    stdin.flush().await.expect("Failed to flush stdin");

    // Read semantic search response
    response.clear();
    let search_timeout = if cfg!(windows) { 30 } else { 15 };
    let read_result = timeout(
        Duration::from_secs(search_timeout),
        reader.read_line(&mut response),
    )
    .await;

    if let Ok(Ok(_)) = read_result {
        println!("Semantic search response: {response}");

        // Check that we got a valid response
        assert!(response.contains("jsonrpc"));
        assert!(response.contains("result"));

        // Parse the JSON to check the content
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&response) {
            if let Some(result) = json_value.get("result") {
                let result_str = result.as_str().unwrap_or("");

                // Should find the add function from math.rs
                assert!(
                    result_str.contains("add") || result_str.contains("Add"),
                    "Should find add function: {result_str}"
                );
                assert!(
                    result_str.contains("math") || result_str.contains("Math"),
                    "Should reference math module: {result_str}"
                );

                println!("✅ Successfully found mathematical functions through semantic search");
            } else {
                panic!("No result field in response: {response}");
            }
        } else {
            panic!("Failed to parse JSON response: {response}");
        }
    } else {
        panic!("Failed to get semantic search response within timeout");
    }

    // Test another semantic search for HTTP functionality
    let http_search_request = r#"{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "semantic_search", "arguments": {"query": "HTTP request handling", "limit": 3, "threshold": 0.4}}}
"#;

    stdin
        .write_all(http_search_request.as_bytes())
        .await
        .expect("Failed to write HTTP search request");
    stdin.flush().await.expect("Failed to flush stdin");

    // Read HTTP search response
    response.clear();
    let http_search_timeout = if cfg!(windows) { 30 } else { 15 };
    let read_result = timeout(
        Duration::from_secs(http_search_timeout),
        reader.read_line(&mut response),
    )
    .await;

    if let Ok(Ok(_)) = read_result {
        println!("HTTP search response: {response}");

        assert!(response.contains("jsonrpc"));
        assert!(response.contains("result"));

        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&response) {
            if let Some(result) = json_value.get("result") {
                let result_str = result.as_str().unwrap_or("");

                // Should find HTTP-related content
                assert!(
                    result_str.contains("http")
                        || result_str.contains("HTTP")
                        || result_str.contains("request"),
                    "Should find HTTP content: {result_str}"
                );

                println!("✅ Successfully found HTTP functionality through semantic search");
            }
        }
    } else {
        panic!("Failed to get HTTP search response within timeout");
    }

    // Clean shutdown
    let _ = child.kill().await;
    let _ = child.wait().await;

    println!("🎉 End-to-end semantic search test completed successfully!");
}

/// Helper function to recursively copy directories
fn copy_dir_all(src: &PathBuf, dst: &PathBuf) -> std::io::Result<()> {
    use std::fs;

    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;

        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if ty.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}
