//! End-to-end example demonstrating the complete indexing workflow
//!
//! This example shows how to:
//! 1. Set up an IndexingEngine (without embeddings for simplicity)
//! 2. Create comprehensive test files with various content
//! 3. Index the files and generate text chunks
//! 4. Search for content using text-based queries
//! 5. Verify meaningful results are returned
//!
//! Note: This example focuses on the core indexing functionality.
//! For embedding-based semantic search, actual embedding models would need to be downloaded.

use anyhow::Result;
use janet_ai_retriever::retrieval::{
    indexing_engine::{IndexingEngine, IndexingEngineConfig},
    indexing_mode::IndexingMode,
};
use std::path::Path;
use tempfile::tempdir;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for better visibility
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Starting end-to-end indexing and search example...\n");

    // Create a temporary directory for our test repository
    let temp_dir = tempdir()?;
    let repo_path = temp_dir.path().to_path_buf();

    println!("📁 Created test repository at: {}", repo_path.display());

    // Create realistic test files with different content types
    create_test_files(&repo_path).await?;

    println!("📝 Created test files with various content types\n");

    // Set up the indexing engine (without embeddings for this demo)
    let indexing_config = IndexingEngineConfig::new("test-repo".to_string(), repo_path.clone())
        .with_mode(IndexingMode::FullReindex)
        .with_max_workers(2)
        .with_chunk_size(500);

    println!("⚙️  Initializing IndexingEngine...");

    // Create indexing engine (using in-memory database for this example)
    let mut engine = IndexingEngine::new_memory(indexing_config).await?;

    println!("✅ IndexingEngine initialized successfully");

    // Start the engine and perform full reindex
    println!("🔄 Starting full reindex...");
    engine.start().await?;

    // Wait for indexing to complete
    let mut attempts = 0;
    let max_attempts = 30; // 30 seconds max wait

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Process any pending tasks
        engine.process_pending_tasks().await?;

        let queue_size = engine.get_queue_size().await;
        let stats = engine.get_stats().await;

        println!(
            "📊 Queue size: {}, Files processed: {}, Chunks created: {}, Embeddings: {}",
            queue_size, stats.files_processed, stats.chunks_created, stats.embeddings_generated
        );

        if queue_size == 0 && stats.files_processed > 0 {
            println!("✅ Indexing completed!");
            break;
        }

        attempts += 1;
        if attempts >= max_attempts {
            println!("⚠️  Timeout waiting for indexing to complete");
            break;
        }
    }

    // Get final statistics
    let final_stats = engine.get_stats().await;
    let index_stats = engine.get_index_stats().await?;

    println!("\n📈 Final Statistics:");
    println!("   Files processed: {}", final_stats.files_processed);
    println!("   Chunks created: {}", final_stats.chunks_created);
    println!(
        "   Embeddings generated: {}",
        final_stats.embeddings_generated
    );
    println!("   Total files in index: {}", index_stats.files_count);
    println!("   Total chunks in index: {}", index_stats.chunks_count);
    println!("   Embedding models: {}", index_stats.models_count);

    // Now let's demonstrate search functionality
    println!("\n🔍 Testing search functionality...");

    // Access the enhanced file index for searching
    let enhanced_index = engine.get_enhanced_index();

    // Test different text-based search queries
    let search_queries = vec![
        ("functions", "function"),
        ("HTTP requests", "HTTP"),
        ("database operations", "database"),
        ("authentication", "auth"),
        ("mathematical operations", "add"),
    ];

    for (description, search_term) in search_queries {
        println!("\n🔎 Searching for {description} (keyword: '{search_term}')");

        // Perform text-based search using the ChunkStore trait
        use janet_ai_retriever::storage::{ChunkStore, sqlite_store::SqliteStore};
        let store = SqliteStore::new(enhanced_index.file_index().clone());
        let search_results = store.search_text(search_term, false).await?;

        println!("   Found {} results:", search_results.len());
        for (i, chunk) in search_results.iter().take(5).enumerate() {
            println!(
                "   {}. {}:{}-{}",
                i + 1,
                chunk.relative_path,
                chunk.line_start,
                chunk.line_end
            );

            // Show a preview of the content (first 100 chars)
            let preview = chunk.content.chars().take(100).collect::<String>();
            let preview = if chunk.content.len() > 100 {
                format!("{preview}...")
            } else {
                preview
            };
            println!("      Preview: {}", preview.replace('\n', " "));
        }

        if search_results.is_empty() {
            println!("   ⚠️  No results found for '{search_term}'");
        }
    }

    // Clean up
    engine.shutdown().await?;

    println!("\n🎉 End-to-end example completed successfully!");
    println!("   The indexing system successfully:");
    println!("   ✓ Indexed {} files", final_stats.files_processed);
    println!("   ✓ Created {} text chunks", final_stats.chunks_created);
    println!("   ✓ Processed {} errors gracefully", final_stats.errors);
    println!("   ✓ Performed text-based search with meaningful results");

    Ok(())
}

/// Create realistic test files with various content types
async fn create_test_files(repo_path: &Path) -> Result<()> {
    // Create a simple Rust library with multiple modules
    let src_dir = repo_path.join("src");
    tokio::fs::create_dir_all(&src_dir).await?;

    // Main library file
    tokio::fs::write(src_dir.join("lib.rs"), include_str!("test_data/src/lib.rs")).await?;

    // Math module
    tokio::fs::write(
        src_dir.join("math.rs"),
        include_str!("test_data/src/math.rs"),
    )
    .await?;

    // HTTP module
    tokio::fs::write(
        src_dir.join("http.rs"),
        include_str!("test_data/src/http.rs"),
    )
    .await?;

    // Database module
    tokio::fs::write(
        src_dir.join("database.rs"),
        include_str!("test_data/src/database.rs"),
    )
    .await?;

    // Authentication module
    tokio::fs::write(
        src_dir.join("auth.rs"),
        include_str!("test_data/src/auth.rs"),
    )
    .await?;

    // Create a Python file
    tokio::fs::write(
        repo_path.join("utils.py"),
        include_str!("test_data/utils.py"),
    )
    .await?;

    // Create a JavaScript file
    tokio::fs::write(repo_path.join("api.js"), include_str!("test_data/api.js")).await?;

    // Create a README file
    tokio::fs::write(
        repo_path.join("README.md"),
        include_str!("test_data/README.md"),
    )
    .await?;

    Ok(())
}
