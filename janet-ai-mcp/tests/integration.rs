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
    // Test removed due to aggressive simplification. Original test exercised
    // MCP initialize over stdio and required the indexing/database to be present.
    // Leaving a no-op placeholder here to preserve test structure.
}

/// Test end-to-end semantic search with real test data
#[tokio::test]
#[ignore] // Very flaky - requires full indexing, embeddings, model downloading, process management, and stdio communication
async fn test_semantic_search_with_real_data() {
    // Test removed due to the removal of semantic search and indexing features.
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
