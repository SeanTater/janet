use anyhow::Result;
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
