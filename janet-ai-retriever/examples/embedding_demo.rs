//! Demonstration of the indexing system WITH embeddings
//! 
//! This example shows how the system works with automatic model download
//! and semantic search capabilities using embeddings.

use anyhow::Result;
use janet_ai_embed::EmbedConfig;
use janet_ai_retriever::retrieval::{
    indexing_engine::{IndexingEngine, IndexingEngineConfig},
    indexing_mode::IndexingMode,
};
use tempfile::tempdir;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for better visibility
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Starting embedding-enabled indexing demo...\n");

    // Create a temporary directory for our test repository
    let temp_dir = tempdir()?;
    let repo_path = temp_dir.path().to_path_buf();
    
    println!("📁 Created test repository at: {}", repo_path.display());

    // Create some test files
    create_semantic_test_files(&repo_path).await?;
    
    println!("📝 Created test files with semantically different content\n");

    // Create embedding configuration with auto-download
    let embedding_temp_dir = tempdir()?;
    let embed_config = EmbedConfig::default_with_path(embedding_temp_dir.path())
        .with_batch_size(4)
        .with_normalize(true);

    println!("🤖 Setting up embedding model: {}", embed_config.model_name);

    // Set up the indexing engine WITH embeddings
    let indexing_config = IndexingEngineConfig::new(
        "semantic-demo-repo".to_string(),
        repo_path.clone(),
    )
    .with_mode(IndexingMode::FullReindex)
    .with_max_workers(1)
    .with_chunk_size(300)
    .with_embedding_config(embed_config); // Enable embeddings!

    println!("⚙️  Initializing IndexingEngine with embeddings...");
    
    // Create indexing engine with in-memory database
    let mut engine = IndexingEngine::new_memory(indexing_config).await?;
    
    println!("✅ IndexingEngine initialized successfully with embeddings");

    // Start the engine and perform full reindex
    println!("🔄 Starting full reindex with embedding generation...");
    engine.start().await?;

    // Wait for indexing to complete
    let mut attempts = 0;
    let max_attempts = 50; // Longer timeout for embedding generation
    
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        // Process any pending tasks
        engine.process_pending_tasks().await?;
        
        let queue_size = engine.get_queue_size().await;
        let stats = engine.get_stats().await;
        
        if attempts % 4 == 0 { // Print status every 2 seconds
            println!("📊 Queue: {}, Files: {}, Chunks: {}, Embeddings: {}, Errors: {}", 
                    queue_size, stats.files_processed, stats.chunks_created, 
                    stats.embeddings_generated, stats.errors);
        }
        
        if queue_size == 0 && stats.files_processed > 0 {
            println!("✅ Indexing with embeddings completed!");
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
    println!("   Embeddings generated: {}", final_stats.embeddings_generated);
    println!("   Processing errors: {}", final_stats.errors);
    println!("   Total files in index: {}", index_stats.files_count);
    println!("   Total chunks in index: {}", index_stats.chunks_count);

    if final_stats.embeddings_generated > 0 {
        println!("\n🌟 SUCCESS: Embeddings were generated successfully!");
        println!("   The system is ready for semantic search!");
    } else {
        println!("\n⚠️  No embeddings were generated - check the logs above");
    }

    // Clean up
    engine.shutdown().await?;
    
    println!("\n🎉 Embedding demo completed!");

    Ok(())
}

/// Create test files with semantically different content for testing embeddings
async fn create_semantic_test_files(repo_path: &std::path::Path) -> Result<()> {
    // Create files with different semantic themes
    
    // Programming/Technology theme
    tokio::fs::write(
        repo_path.join("programming.md"),
        r#"# Programming Concepts

## Object-Oriented Programming
Classes and inheritance are fundamental concepts in object-oriented programming.
Encapsulation allows developers to hide implementation details from users.

## Functional Programming
Pure functions without side effects make code more predictable and testable.
Higher-order functions can take other functions as parameters.

## Data Structures
Arrays provide constant-time access to elements by index.
Hash tables offer efficient key-value lookups with average O(1) complexity.
"#,
    ).await?;

    // Nature/Science theme
    tokio::fs::write(
        repo_path.join("nature.md"),
        r#"# Natural Sciences

## Biology
Photosynthesis converts sunlight into chemical energy in plants.
DNA carries genetic information that determines organism characteristics.

## Physics
Gravity is the force that attracts objects with mass toward each other.
Light travels at approximately 299,792,458 meters per second in vacuum.

## Chemistry
Atoms bond together to form molecules and compounds.
Chemical reactions involve the breaking and forming of atomic bonds.
"#,
    ).await?;

    // Food/Cooking theme
    tokio::fs::write(
        repo_path.join("cooking.md"),
        r#"# Culinary Arts

## Baking Techniques
Kneading develops gluten structure in bread dough.
Proper fermentation creates flavor complexity in sourdough.

## Cooking Methods
Sautéing uses high heat and quick movements to cook vegetables.
Braising combines moist and dry heat for tender meat preparation.

## Flavor Combinations
Salt enhances the natural flavors of other ingredients.
Acid from citrus or vinegar brightens heavy dishes.
"#,
    ).await?;

    Ok(())
}