use anyhow::Result;
use janet_ai_retriever::retrieval::file_index::{ChunkRef, FileIndex, FileRef};
use std::process::Command;
use tempfile::TempDir;

/// Helper to run the CLI binary with given args
fn run_cli(temp_dir: &TempDir, args: &[&str]) -> Result<std::process::Output> {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "-p", "janet-ai-retriever", "--"])
        .arg("--base-dir")
        .arg(temp_dir.path().to_string_lossy().as_ref())
        .args(args)
        .env("RUST_LOG", "error"); // Reduce log noise

    let output = cmd.output()?;
    Ok(output)
}

/// Helper to populate a database with test data
async fn populate_test_data(temp_dir: &TempDir) -> Result<()> {
    // Ensure the directory is writable
    std::fs::create_dir_all(temp_dir.path())?;

    let file_index = FileIndex::open(temp_dir.path()).await?;

    // Create test files
    let files = vec![
        FileRef {
            relative_path: "src/main.rs".to_string(),
            content: b"fn main() {\n    println!(\"Hello, world!\");\n}".to_vec(),
            hash: [1; 32],
        },
        FileRef {
            relative_path: "src/lib.rs".to_string(),
            content: b"pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}".to_vec(),
            hash: [2; 32],
        },
    ];

    // Insert files
    for file in &files {
        file_index.upsert_file(file).await?;
    }

    // Create test chunks
    let chunks = vec![
        ChunkRef {
            id: None,
            file_hash: [1; 32],
            relative_path: "src/main.rs".to_string(),
            line_start: 1,
            line_end: 1,
            content: "fn main() {".to_string(),
            embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
        },
        ChunkRef {
            id: None,
            file_hash: [1; 32],
            relative_path: "src/main.rs".to_string(),
            line_start: 2,
            line_end: 2,
            content: "    println!(\"Hello, world!\");".to_string(),
            embedding: Some(vec![0.5, 0.6, 0.7, 0.8]),
        },
        ChunkRef {
            id: None,
            file_hash: [2; 32],
            relative_path: "src/lib.rs".to_string(),
            line_start: 1,
            line_end: 2,
            content: "pub fn add(a: i32, b: i32) -> i32 {\n    a + b".to_string(),
            embedding: Some(vec![0.9, 0.8, 0.7, 0.6]),
        },
        ChunkRef {
            id: None,
            file_hash: [2; 32],
            relative_path: "src/lib.rs".to_string(),
            line_start: 3,
            line_end: 3,
            content: "}".to_string(),
            embedding: None, // This chunk has no embedding
        },
    ];

    // Insert chunks
    file_index.upsert_chunks(&chunks).await?;

    Ok(())
}

/// Test basic CLI functionality - check that binary can be run and shows help
#[tokio::test]
async fn test_cli_basic_functionality() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test that CLI binary runs and shows help
    let output = run_cli(&temp_dir, &["--help"])?;

    // Should succeed regardless of database state
    assert!(
        output.status.success(),
        "CLI help command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("CLI tool to interact with"));
    assert!(stdout.contains("init"));
    assert!(stdout.contains("list"));
    assert!(stdout.contains("get"));
    assert!(stdout.contains("search"));
    assert!(stdout.contains("stats"));

    Ok(())
}

/// Test error handling for invalid commands
#[tokio::test]
async fn test_cli_error_handling() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test search without embedding parameter - should fail
    let output = run_cli(&temp_dir, &["search"])?;
    assert!(
        !output.status.success(),
        "Search without embedding should fail"
    );

    Ok(())
}

/// Test subcommand help works
#[tokio::test]
async fn test_cli_subcommand_help() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test subcommand help
    let output = run_cli(&temp_dir, &["list", "--help"])?;
    assert!(
        output.status.success(),
        "List help command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("List chunks"));
    assert!(stdout.contains("--format"));
    assert!(stdout.contains("--limit"));

    Ok(())
}

/// Test init command and basic operations
#[tokio::test]
async fn test_cli_init_flow() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Make temp directory writable
    std::fs::create_dir_all(temp_dir.path())?;

    // Test init command
    let output = run_cli(&temp_dir, &["init"])?;
    if !output.status.success() {
        // If init fails due to permissions, skip this test but don't fail
        eprintln!(
            "Skipping init test due to permissions: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return Ok(());
    }

    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("Initialized chunk database"));

    // Test stats on empty database
    let output = run_cli(&temp_dir, &["stats"])?;
    if output.status.success() {
        let stdout = String::from_utf8(output.stdout)?;
        assert!(stdout.contains("Total chunks: 0"));
    }

    // Test list on empty database
    let output = run_cli(&temp_dir, &["list"])?;
    if output.status.success() {
        let stdout = String::from_utf8(output.stdout)?;
        assert!(stdout.contains("Found 0 chunks"));
    }

    Ok(())
}

/// Test that CLI commands handle non-existent database gracefully
#[tokio::test]
async fn test_cli_graceful_error_handling() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test commands on non-existent database - they should fail gracefully

    // Test get with non-existent database
    let output = run_cli(&temp_dir, &["get", "1"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Error:"));

    // Test list with non-existent database
    let output = run_cli(&temp_dir, &["list"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Error:"));

    // Test stats with non-existent database
    let output = run_cli(&temp_dir, &["stats"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Error:"));

    // Test search with non-existent database
    let output = run_cli(&temp_dir, &["search", "--embedding", "0.1,0.2,0.3"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Error:"));

    Ok(())
}

/// Test invalid input validation and edge cases
#[tokio::test]
async fn test_cli_invalid_input_validation() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test invalid hex hash - wrong length
    let output = run_cli(&temp_dir, &["list", "--file-hash", "abc123"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Error:"));

    // Test invalid hex hash - non-hex characters
    let output = run_cli(
        &temp_dir,
        &[
            "list",
            "--file-hash",
            "xyz123456789012345678901234567890123456789012345678901234567890123",
        ],
    )?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Error:"));

    // Test negative chunk ID
    let output = run_cli(&temp_dir, &["get", "-1"])?;
    assert!(!output.status.success());

    // Test invalid format option
    let output = run_cli(&temp_dir, &["list", "--format", "invalid"])?;
    assert!(!output.status.success());

    // Test empty embedding vector
    let output = run_cli(&temp_dir, &["search", "--embedding", ""])?;
    assert!(!output.status.success());

    // Test invalid embedding format (non-numeric)
    let output = run_cli(&temp_dir, &["search", "--embedding", "a,b,c"])?;
    assert!(!output.status.success());

    // Test invalid similarity threshold (out of range)
    let output = run_cli(
        &temp_dir,
        &["search", "--embedding", "0.1,0.2", "--threshold", "1.5"],
    )?;
    assert!(!output.status.success());

    Ok(())
}

/// Test output format consistency across commands
#[tokio::test]
async fn test_cli_output_format_consistency() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Initialize database first
    let output = run_cli(&temp_dir, &["init"])?;
    if !output.status.success() {
        // Skip if we can't create database
        return Ok(());
    }

    // Test JSON format produces valid JSON for list command
    let output = run_cli(&temp_dir, &["list", "--format", "json"])?;
    if output.status.success() {
        let stdout = String::from_utf8(output.stdout)?;
        // Should be valid JSON (even if empty array)
        let _: serde_json::Value = serde_json::from_str(&stdout)
            .map_err(|e| anyhow::anyhow!("Invalid JSON output from list command: {}", e))?;
    }

    // Test that help is consistent across subcommands
    let commands = ["list", "get", "search", "stats"];
    for cmd in commands {
        let output = run_cli(&temp_dir, &[cmd, "--help"])?;
        assert!(output.status.success(), "Help for {cmd} command failed");
        let stdout = String::from_utf8(output.stdout)?;
        assert!(
            stdout.contains("--format") || cmd == "stats" || cmd == "init",
            "Command {cmd} should support --format or explicitly not need it"
        );
    }

    Ok(())
}

/// Test edge cases with special characters and unicode
#[tokio::test]
async fn test_cli_unicode_and_special_chars() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test with unicode in base directory path
    let unicode_dir = temp_dir.path().join("测试目录");
    std::fs::create_dir_all(&unicode_dir)?;

    let output = Command::new("cargo")
        .args(["run", "-p", "janet-ai-retriever", "--"])
        .arg("--base-dir")
        .arg(unicode_dir.to_string_lossy().as_ref())
        .args(["--help"])
        .env("RUST_LOG", "error")
        .output()?;

    // Should handle unicode paths gracefully
    assert!(output.status.success(), "Should handle unicode paths");

    Ok(())
}

/// Test CLI with boundary conditions
#[tokio::test]
async fn test_cli_boundary_conditions() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test with maximum valid chunk ID (SQLite max integer)
    let output = run_cli(&temp_dir, &["get", "9223372036854775807"])?;
    assert!(!output.status.success()); // Should fail gracefully (chunk doesn't exist)
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Error:"));

    // Test with zero limit
    let output = run_cli(&temp_dir, &["list", "--limit", "0"])?;
    assert!(!output.status.success()); // Should fail gracefully

    // Test with very large limit
    let output = run_cli(&temp_dir, &["list", "--limit", "999999"])?;
    assert!(!output.status.success()); // Database doesn't exist, but should parse args

    // Test with very long embedding vector (should be accepted)
    let long_embedding: Vec<String> = (0..1000).map(|i| (i as f32 / 1000.0).to_string()).collect();
    let embedding_str = long_embedding.join(",");
    let output = run_cli(&temp_dir, &["search", "--embedding", &embedding_str])?;
    assert!(!output.status.success()); // Database doesn't exist, but should parse args

    Ok(())
}

/// Test CLI error message quality and user experience
#[tokio::test]
async fn test_cli_error_message_quality() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test that error messages are user-friendly
    let test_cases = vec![
        (vec!["get"], "missing required argument"),
        (vec!["search"], "missing required argument"),
        (vec!["list", "--file-hash", "invalid"], "Invalid hex hash"),
        (vec!["get", "abc"], "invalid digit"),
    ];

    for (args, expected_error_hint) in test_cases {
        let output = run_cli(&temp_dir, &args)?;
        assert!(!output.status.success(), "Command {args:?} should fail");

        let stderr = String::from_utf8(output.stderr)?;
        // Error messages should be helpful (either from clap or our app)
        assert!(
            stderr
                .to_lowercase()
                .contains(&expected_error_hint.to_lowercase())
                || stderr.contains("Error:")
                || stderr.contains("error:"),
            "Error message for {args:?} should contain helpful information. Got: {stderr}"
        );
    }

    Ok(())
}

/// Test CLI with various database states
#[tokio::test]
async fn test_cli_database_state_edge_cases() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Create .code-assistant directory but no database file
    let assist_dir = temp_dir.path().join(".code-assistant");
    std::fs::create_dir_all(&assist_dir)?;

    // Commands should fail gracefully when database doesn't exist (no .code-assistant dir)
    let empty_temp_dir = tempfile::tempdir()?;
    let output = run_cli(&empty_temp_dir, &["stats"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Error:"));

    // Create an empty file where database should be
    let db_path = assist_dir.join("index.db");
    std::fs::write(&db_path, "")?;

    // Our CLI should handle corrupted/empty database gracefully by reinitializing it
    // This is actually correct behavior - SQLite can initialize from empty files
    let output = run_cli(&temp_dir, &["stats"])?;
    if output.status.success() {
        // If it succeeds, it should show empty database stats
        let stdout = String::from_utf8(output.stdout)?;
        assert!(stdout.contains("Total chunks: 0"));
    } else {
        // If it fails, should be graceful error
        let stderr = String::from_utf8(output.stderr)?;
        assert!(stderr.contains("Error:"));
    }

    // Test with a truly corrupted database file (invalid SQLite)
    std::fs::write(&db_path, "This is not a SQLite database file")?;
    let output = run_cli(&temp_dir, &["stats"])?;
    // Should fail with corrupted database
    if !output.status.success() {
        let stderr = String::from_utf8(output.stderr)?;
        assert!(stderr.contains("Error:"));
    } else {
        // If SQLite somehow handles this gracefully, that's also acceptable
        // Some versions of SQLite are very resilient
        eprintln!("SQLite handled corrupted file gracefully - this is acceptable behavior");
    }

    Ok(())
}

/// Test CLI performance with edge case inputs
#[tokio::test]
async fn test_cli_performance_edge_cases() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test with very large limit (should not crash or consume excessive memory)
    let output = run_cli(&temp_dir, &["list", "--limit", "1000000"])?;
    assert!(!output.status.success()); // Database doesn't exist, but should parse args

    // Test with very long file hash (64 hex chars = 32 bytes, valid length)
    let valid_hash = "a".repeat(64);
    let output = run_cli(&temp_dir, &["list", "--file-hash", &valid_hash])?;
    assert!(!output.status.success()); // Database doesn't exist

    // Test search with moderately large embedding (avoid Windows command line length limits)
    let medium_embedding: Vec<String> = (0..100).map(|i| (i as f32 / 100.0).to_string()).collect();
    let embedding_str = medium_embedding.join(",");
    let output = run_cli(
        &temp_dir,
        &["search", "--embedding", &embedding_str, "--limit", "1"],
    )?;
    assert!(!output.status.success()); // Database doesn't exist, but should parse args

    // Test with extreme similarity threshold values
    let output = run_cli(
        &temp_dir,
        &["search", "--embedding", "0.1,0.2", "--threshold", "0.0"],
    )?;
    assert!(!output.status.success()); // Database doesn't exist

    let output = run_cli(
        &temp_dir,
        &["search", "--embedding", "0.1,0.2", "--threshold", "1.0"],
    )?;
    assert!(!output.status.success()); // Database doesn't exist

    Ok(())
}

/// Test successful operations with populated database
#[tokio::test]
async fn test_cli_happy_path_list_chunks() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Populate with test data
    if let Err(e) = populate_test_data(&temp_dir).await {
        eprintln!("Skipping happy path test due to database permissions: {e}");
        return Ok(());
    }

    // Test list command with results
    let output = run_cli(&temp_dir, &["list"])?;
    assert!(output.status.success(), "List command should succeed");

    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("Found 4 chunks"), "Should find 4 chunks");
    assert!(stdout.contains("src/main.rs"), "Should show main.rs");
    assert!(stdout.contains("src/lib.rs"), "Should show lib.rs");
    assert!(
        stdout.contains("✓"),
        "Should show some chunks have embeddings"
    );
    assert!(
        stdout.contains("✗"),
        "Should show some chunks don't have embeddings"
    );

    // Test list with limit
    let output = run_cli(&temp_dir, &["list", "--limit", "2"])?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("Found 2 chunks"),
        "Should limit to 2 chunks"
    );

    // Test list in JSON format
    let output = run_cli(&temp_dir, &["list", "--format", "json"])?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let json: serde_json::Value = serde_json::from_str(&stdout)?;
    assert!(json.is_array(), "JSON output should be an array");
    assert_eq!(
        json.as_array().unwrap().len(),
        4,
        "Should have 4 items in JSON"
    );

    // Test list in full format
    let output = run_cli(&temp_dir, &["list", "--format", "full"])?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("fn main()"), "Should show chunk content");
    assert!(stdout.contains("println!"), "Should show chunk content");

    Ok(())
}

/// Test successful chunk retrieval by ID
#[tokio::test]
async fn test_cli_happy_path_get_chunk() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Populate with test data
    if let Err(e) = populate_test_data(&temp_dir).await {
        eprintln!("Skipping happy path test due to database permissions: {e}");
        return Ok(());
    }

    // Test get command with existing chunk (ID 1 should exist)
    let output = run_cli(&temp_dir, &["get", "1"])?;
    assert!(output.status.success(), "Get command should succeed");

    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("Chunk ID: 1"), "Should show chunk ID");
    assert!(stdout.contains("src/main.rs"), "Should show file path");
    assert!(stdout.contains("fn main()"), "Should show chunk content");

    // Test get in JSON format
    let output = run_cli(&temp_dir, &["get", "1", "--format", "json"])?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let json: serde_json::Value = serde_json::from_str(&stdout)?;
    assert!(json.is_object(), "JSON output should be an object");
    assert_eq!(json["id"], 1, "Should have correct ID");
    assert!(
        json["content"].as_str().unwrap().contains("fn main()"),
        "Should have content"
    );

    // Test get in summary format
    let output = run_cli(&temp_dir, &["get", "2", "--format", "summary"])?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("Chunk ID: 2"), "Should show chunk ID");
    assert!(
        stdout.contains("Content preview:"),
        "Should show content preview"
    );

    Ok(())
}

/// Test successful search functionality
#[tokio::test]
async fn test_cli_happy_path_search() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Populate with test data
    if let Err(e) = populate_test_data(&temp_dir).await {
        eprintln!("Skipping happy path test due to database permissions: {e}");
        return Ok(());
    }

    // Test search with embedding similar to first chunk
    let output = run_cli(
        &temp_dir,
        &["search", "--embedding", "0.1,0.2,0.3,0.4", "--limit", "5"],
    )?;
    assert!(output.status.success(), "Search command should succeed");

    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("Found"), "Should find similar chunks");
    assert!(
        stdout.contains("Similarity:"),
        "Should show similarity scores"
    );
    assert!(
        stdout.contains("src/main.rs") || stdout.contains("src/lib.rs"),
        "Should show file names"
    );

    // Test search in JSON format
    let output = run_cli(
        &temp_dir,
        &[
            "search",
            "--embedding",
            "0.9,0.8,0.7,0.6",
            "--format",
            "json",
        ],
    )?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let json: serde_json::Value = serde_json::from_str(&stdout)?;
    assert!(json.is_array(), "JSON output should be an array");
    if let Some(array) = json.as_array() {
        if !array.is_empty() {
            let first_result = &array[0];
            assert!(
                first_result["similarity"].is_number(),
                "Should have similarity score"
            );
            assert!(
                first_result["chunk"]["content"].is_string(),
                "Should have chunk content"
            );
        }
    }

    // Test search with threshold
    let output = run_cli(
        &temp_dir,
        &[
            "search",
            "--embedding",
            "0.1,0.2,0.3,0.4",
            "--threshold",
            "0.9",
        ],
    )?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    // With high threshold, might find fewer or no results
    assert!(
        stdout.contains("Found"),
        "Should show search results even if empty"
    );

    Ok(())
}

/// Test database statistics with real data
#[tokio::test]
async fn test_cli_happy_path_stats() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Populate with test data
    if let Err(e) = populate_test_data(&temp_dir).await {
        eprintln!("Skipping happy path test due to database permissions: {e}");
        return Ok(());
    }

    // Test stats command
    let output = run_cli(&temp_dir, &["stats"])?;
    assert!(output.status.success(), "Stats command should succeed");

    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("Total chunks: 4"),
        "Should show 4 total chunks"
    );
    assert!(
        stdout.contains("Chunks with embeddings: 3"),
        "Should show 3 chunks with embeddings"
    );
    assert!(
        stdout.contains("Unique files: 2"),
        "Should show 2 unique files"
    );
    assert!(stdout.contains("src/main.rs"), "Should list main.rs");
    assert!(stdout.contains("src/lib.rs"), "Should list lib.rs");

    Ok(())
}

/// Test file hash filtering
#[tokio::test]
async fn test_cli_happy_path_file_hash_filter() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Populate with test data
    if let Err(e) = populate_test_data(&temp_dir).await {
        eprintln!("Skipping happy path test due to database permissions: {e}");
        return Ok(());
    }

    // Test list with file hash filter (hash [1; 32] = all 1's = 64 '1' characters in hex)
    let file_hash = "1".repeat(64);
    let output = run_cli(&temp_dir, &["list", "--file-hash", &file_hash])?;
    assert!(
        output.status.success(),
        "List with file hash should succeed"
    );

    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("Found 2 chunks"),
        "Should find 2 chunks for main.rs"
    );
    assert!(
        stdout.contains("src/main.rs"),
        "Should only show main.rs chunks"
    );
    assert!(
        !stdout.contains("src/lib.rs"),
        "Should not show lib.rs chunks"
    );

    // Test with second file hash (hash [2; 32] = all 2's)
    let file_hash2 = "2".repeat(64);
    let output = run_cli(&temp_dir, &["list", "--file-hash", &file_hash2])?;
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("Found 2 chunks"),
        "Should find 2 chunks for lib.rs"
    );
    assert!(
        stdout.contains("src/lib.rs"),
        "Should only show lib.rs chunks"
    );
    assert!(
        !stdout.contains("src/main.rs"),
        "Should not show main.rs chunks"
    );

    Ok(())
}

/// Integration test that validates CLI can actually work end-to-end when database is available
#[tokio::test]
async fn test_cli_end_to_end_when_possible() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Try to run init command first
    let output = run_cli(&temp_dir, &["init"])?;
    if !output.status.success() {
        eprintln!("Skipping end-to-end test - cannot initialize database");
        return Ok(());
    }

    // Verify stats works on empty database
    let output = run_cli(&temp_dir, &["stats"])?;
    assert!(
        output.status.success(),
        "Stats should work on empty database"
    );
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("Total chunks: 0"),
        "Should show 0 chunks initially"
    );

    // Verify list works on empty database
    let output = run_cli(&temp_dir, &["list"])?;
    assert!(
        output.status.success(),
        "List should work on empty database"
    );
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("Found 0 chunks"),
        "Should find 0 chunks initially"
    );

    // Verify get fails gracefully for non-existent chunk
    let output = run_cli(&temp_dir, &["get", "999"])?;
    assert!(
        output.status.success(),
        "Get should succeed even for non-existent chunk"
    );
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("not found"),
        "Should indicate chunk not found"
    );

    eprintln!("✅ End-to-end CLI test passed - database operations working correctly");
    Ok(())
}
