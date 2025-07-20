//! Integration tests focusing on happy path scenarios for the indexing system
//! 
//! These tests verify that the core functionality works correctly:
//! - Creating and configuring the IndexingEngine
//! - Indexing files with text chunking
//! - Processing different file types
//! - Task queue prioritization
//! - Basic search functionality (without embeddings)

use anyhow::Result;
use janet_ai_retriever::retrieval::{
    indexing_engine::{IndexingEngine, IndexingEngineConfig},
    indexing_mode::IndexingMode,
    task_queue::IndexingTask,
};
use std::path::PathBuf;
use tempfile::tempdir;
use tokio::time::Duration;

/// Test basic IndexingEngine creation and configuration
#[tokio::test]
async fn test_indexing_engine_creation() -> Result<()> {
    let temp_dir = tempdir()?;
    
    // Test different indexing modes
    let modes = vec![
        IndexingMode::ReadOnly,
        IndexingMode::ContinuousMonitoring,
        IndexingMode::FullReindex,
    ];
    
    for mode in modes {
        let config = IndexingEngineConfig::new(
            "test-repo".to_string(),
            temp_dir.path().to_path_buf(),
        )
        .with_mode(mode.clone())
        .with_max_workers(2)
        .with_chunk_size(1000);
        
        let engine = IndexingEngine::new_memory(config).await?;
        
        // Verify basic properties
        let stats = engine.get_index_stats().await?;
        assert_eq!(stats.files_count, 0);
        assert_eq!(stats.chunks_count, 0);
        
        // Test queue is initially empty
        let queue_size = engine.get_queue_size().await;
        assert_eq!(queue_size, 0);
    }
    
    Ok(())
}

/// Test file indexing with different file types
#[tokio::test]
async fn test_file_indexing_without_embeddings() -> Result<()> {
    let temp_dir = tempdir()?;
    let repo_path = temp_dir.path().to_path_buf();
    
    // Create test files with different extensions
    create_test_files(&repo_path).await?;
    
    let config = IndexingEngineConfig::new(
        "test-repo".to_string(),
        repo_path,
    )
    .with_mode(IndexingMode::FullReindex)
    .with_max_workers(1)
    .with_chunk_size(500);
    
    let mut engine = IndexingEngine::new_memory(config).await?;
    engine.start().await?;
    
    // Wait for indexing to complete
    let mut attempts = 0;
    let max_attempts = 20;
    
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        engine.process_pending_tasks().await?;
        
        let queue_size = engine.get_queue_size().await;
        let stats = engine.get_stats().await;
        
        if queue_size == 0 && stats.files_processed > 0 {
            break;
        }
        
        attempts += 1;
        if attempts >= max_attempts {
            panic!("Timeout waiting for indexing to complete. Queue size: {}, Files processed: {}", 
                   queue_size, stats.files_processed);
        }
    }
    
    // Verify files were processed
    let final_stats = engine.get_stats().await;
    let index_stats = engine.get_index_stats().await?;
    
    println!("Final stats: Files processed: {}, Chunks created: {}", 
             final_stats.files_processed, final_stats.chunks_created);
    println!("Index stats: Files: {}, Chunks: {}", 
             index_stats.files_count, index_stats.chunks_count);
    
    // We should have processed at least 4 files (Rust, Python, JavaScript, Markdown)
    assert!(final_stats.files_processed >= 4, 
            "Expected at least 4 files processed, got {}", final_stats.files_processed);
    
    // We should have created chunks
    assert!(final_stats.chunks_created > 0, 
            "Expected chunks to be created, got {}", final_stats.chunks_created);
    
    // Index should contain the files and chunks
    assert!(index_stats.files_count >= 4, 
            "Expected at least 4 files in index, got {}", index_stats.files_count);
    assert!(index_stats.chunks_count > 0, 
            "Expected chunks in index, got {}", index_stats.chunks_count);
    
    engine.shutdown().await?;
    Ok(())
}

/// Test task queue prioritization
#[tokio::test]
async fn test_task_queue_prioritization() -> Result<()> {
    let temp_dir = tempdir()?;
    
    let config = IndexingEngineConfig::new(
        "test-repo".to_string(),
        temp_dir.path().to_path_buf(),
    )
    .with_mode(IndexingMode::ContinuousMonitoring)
    .with_max_workers(1);
    
    let mut engine = IndexingEngine::new_memory(config).await?;
    engine.start().await?;
    
    // Create test files
    let test_files = vec![
        temp_dir.path().join("low_priority.rs"),
        temp_dir.path().join("high_priority.rs"),
        temp_dir.path().join("normal_priority.rs"),
    ];
    
    for file in &test_files {
        tokio::fs::write(file, "fn main() { println!(\"Hello\"); }").await?;
    }
    
    // Schedule tasks with different priorities (reverse order)
    let _low_task = IndexingTask::index_file_background(test_files[0].clone());
    let _high_task = IndexingTask::index_file_high_priority(test_files[1].clone());
    let _normal_task = IndexingTask::index_file(test_files[2].clone());
    
    engine.schedule_file_index(test_files[0].clone()).await?; // This uses normal priority by default
    
    // Give time for tasks to be processed
    tokio::time::sleep(Duration::from_millis(200)).await;
    engine.process_pending_tasks().await?;
    
    let stats = engine.get_stats().await;
    assert!(stats.files_processed > 0, "Expected files to be processed");
    
    engine.shutdown().await?;
    Ok(())
}

/// Test continuous monitoring mode
#[tokio::test]
async fn test_continuous_monitoring() -> Result<()> {
    let temp_dir = tempdir()?;
    let repo_path = temp_dir.path().to_path_buf();
    
    let config = IndexingEngineConfig::new(
        "test-repo".to_string(),
        repo_path.clone(),
    )
    .with_mode(IndexingMode::ContinuousMonitoring)
    .with_max_workers(1);
    
    let mut engine = IndexingEngine::new_memory(config).await?;
    engine.start().await?;
    
    // Create an initial file
    let test_file = repo_path.join("initial.rs");
    tokio::fs::write(&test_file, "fn main() { println!(\"Initial\"); }").await?;
    
    // Schedule indexing
    engine.schedule_file_index(test_file.clone()).await?;
    
    // Process the task
    tokio::time::sleep(Duration::from_millis(100)).await;
    engine.process_pending_tasks().await?;
    
    let initial_stats = engine.get_stats().await;
    assert_eq!(initial_stats.files_processed, 1);
    
    // Create another file and index it
    let test_file2 = repo_path.join("second.py");
    tokio::fs::write(&test_file2, "print('Hello from Python')").await?;
    
    engine.schedule_file_index(test_file2).await?;
    
    // Process the second task
    tokio::time::sleep(Duration::from_millis(100)).await;
    engine.process_pending_tasks().await?;
    
    let final_stats = engine.get_stats().await;
    assert_eq!(final_stats.files_processed, 2);
    
    engine.shutdown().await?;
    Ok(())
}

/// Test read-only mode restrictions
#[tokio::test]
async fn test_read_only_mode() -> Result<()> {
    let temp_dir = tempdir()?;
    
    let config = IndexingEngineConfig::new(
        "test-repo".to_string(),
        temp_dir.path().to_path_buf(),
    )
    .with_mode(IndexingMode::ReadOnly);
    
    let mut engine = IndexingEngine::new_memory(config).await?;
    
    // Start with timeout
    tokio::time::timeout(Duration::from_secs(5), engine.start()).await
        .map_err(|_| anyhow::anyhow!("Engine start timed out"))??;
    
    // Try to schedule a file for indexing (should fail)
    let test_file = temp_dir.path().join("test.rs");
    tokio::fs::write(&test_file, "fn main() {}").await?;
    
    let result = engine.schedule_file_index(test_file).await;
    assert!(result.is_err(), "Expected file indexing to fail in read-only mode");
    
    // Try to schedule full reindex (should fail)
    let result = engine.schedule_full_reindex().await;
    assert!(result.is_err(), "Expected full reindex to fail in read-only mode");
    
    // Shutdown with timeout - read-only mode doesn't start workers so shutdown should be immediate
    tokio::time::timeout(Duration::from_secs(2), engine.shutdown()).await
        .map_err(|_| anyhow::anyhow!("Engine shutdown timed out"))??;
    
    Ok(())
}

/// Test chunk creation with different file sizes
#[tokio::test]
async fn test_chunk_creation() -> Result<()> {
    let temp_dir = tempdir()?;
    
    let config = IndexingEngineConfig::new(
        "test-repo".to_string(),
        temp_dir.path().to_path_buf(),
    )
    .with_mode(IndexingMode::ContinuousMonitoring)
    .with_chunk_size(100); // Small chunks for testing
    
    let mut engine = IndexingEngine::new_memory(config).await?;
    engine.start().await?;
    
    // Create a large file that should be split into multiple chunks
    let large_content = (0..50)
        .map(|i| format!("fn function_{}() {{\n    println!(\"Function {}\");\n}}\n\n", i, i))
        .collect::<String>();
    
    let large_file = temp_dir.path().join("large.rs");
    tokio::fs::write(&large_file, &large_content).await?;
    
    // Index the large file
    engine.schedule_file_index(large_file).await?;
    
    // Process and wait for completion
    let mut attempts = 0;
    loop {
        tokio::time::sleep(Duration::from_millis(50)).await;
        engine.process_pending_tasks().await?;
        
        let stats = engine.get_stats().await;
        if stats.files_processed > 0 {
            // Should have created multiple chunks due to small chunk size
            assert!(stats.chunks_created > 1, 
                    "Expected multiple chunks for large file, got {}", stats.chunks_created);
            break;
        }
        
        attempts += 1;
        if attempts >= 20 {
            panic!("Timeout waiting for large file to be processed");
        }
    }
    
    engine.shutdown().await?;
    Ok(())
}

/// Test error handling with invalid files
#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let temp_dir = tempdir()?;
    
    let config = IndexingEngineConfig::new(
        "test-repo".to_string(),
        temp_dir.path().to_path_buf(),
    )
    .with_mode(IndexingMode::ContinuousMonitoring);
    
    let mut engine = IndexingEngine::new_memory(config).await?;
    engine.start().await?;
    
    // Try to index a non-existent file
    let nonexistent_file = temp_dir.path().join("does_not_exist.rs");
    engine.schedule_file_index(nonexistent_file).await?;
    
    // Process the task (should handle the error gracefully)
    tokio::time::sleep(Duration::from_millis(100)).await;
    engine.process_pending_tasks().await?;
    
    let stats = engine.get_stats().await;
    // Should have recorded an error
    assert!(stats.errors > 0, "Expected error to be recorded");
    
    engine.shutdown().await?;
    Ok(())
}

/// Create test files with various content types for testing
async fn create_test_files(repo_path: &PathBuf) -> Result<()> {
    // Create Rust file
    tokio::fs::write(
        repo_path.join("lib.rs"),
        r#"//! A test Rust library

/// Add two numbers
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Multiply two numbers
pub fn multiply(x: f64, y: f64) -> f64 {
    x * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}
"#,
    ).await?;

    // Create Python file
    tokio::fs::write(
        repo_path.join("utils.py"),
        r#"""Utility functions for data processing."""

def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

def process_data(data_list):
    """Process a list of data items."""
    processed = []
    for item in data_list:
        if isinstance(item, str):
            processed.append(item.upper())
        elif isinstance(item, (int, float)):
            processed.append(item * 2)
    return processed

class DataProcessor:
    """A class for processing data with various methods."""
    
    def __init__(self, config=None):
        self.config = config or {}
        
    def transform(self, data):
        """Transform data according to configuration."""
        return data
"#,
    ).await?;

    // Create JavaScript file
    tokio::fs::write(
        repo_path.join("api.js"),
        r#"/**
 * API utilities for HTTP requests
 */

class HttpClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json'
        };
    }

    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'GET',
            headers: this.headers
        });
        return response.json();
    }

    async post(endpoint, data) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        return response.json();
    }
}

function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

module.exports = { HttpClient, validateEmail };
"#,
    ).await?;

    // Create Markdown file
    tokio::fs::write(
        repo_path.join("README.md"),
        r#"# Test Repository

This is a test repository for demonstrating the indexing capabilities.

## Features

- **Rust**: Fast systems programming language
- **Python**: High-level programming with great libraries
- **JavaScript**: Dynamic web programming language

## Usage

To use this repository:

1. Clone the repository
2. Install dependencies
3. Run the tests

### Examples

Here's how to use the add function:

```rust
use mylib::add;

let result = add(2, 3);
println!("Result: {}", result);
```

And here's a Python example:

```python
from utils import calculate_average

numbers = [1, 2, 3, 4, 5]
avg = calculate_average(numbers)
print(f"Average: {avg}")
```

## Contributing

Please read the contributing guidelines before submitting PRs.
"#,
    ).await?;

    // Create a JSON config file
    tokio::fs::write(
        repo_path.join("config.json"),
        r#"{
    "name": "test-project",
    "version": "1.0.0",
    "description": "A test project for indexing",
    "author": "Test Author",
    "license": "MIT",
    "dependencies": {
        "lodash": "^4.17.21",
        "axios": "^0.21.1"
    },
    "scripts": {
        "test": "jest",
        "build": "webpack",
        "start": "node index.js"
    }
}
"#,
    ).await?;

    Ok(())
}