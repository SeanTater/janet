//! Simple example demonstrating real embedding generation with fastembed

use janet_ai_embed::{EmbedConfig, EmbeddingProvider, FastEmbedProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🤖 Janet AI Embed - Real Embedding Example");
    println!("==========================================");

    // Create configuration for a built-in model
    let temp_dir = tempfile::tempdir()?;
    let config = EmbedConfig::default_with_path(temp_dir.path())
        .with_batch_size(2)
        .with_normalize(true);

    println!("📝 Creating FastEmbed provider with config:");
    println!("   Model: {}", config.model_name);
    println!("   Batch size: {}", config.batch_size);
    println!("   Normalize: {}", config.normalize);

    // Create and initialize the provider
    let provider = FastEmbedProvider::create(config).await?;

    println!("✅ Provider initialized successfully!");
    println!("   Dimension: {}", provider.embedding_dimension());
    println!("   Provider: {}", provider.provider_name());

    // Generate embedding for a single text
    println!("\n📊 Generating embedding for single text...");
    let text = "Hello, this is a test sentence for embedding generation.";
    let embedding = provider.embed_text(text).await?;

    println!("   Text: \"{text}\"");
    println!("   Embedding dimension: {}", embedding.len());
    println!(
        "   First 5 values: {:?}",
        &embedding[..5.min(embedding.len())]
    );

    // Generate embeddings for multiple texts
    println!("\n📊 Generating embeddings for multiple texts...");
    let texts = vec![
        "Rust is a systems programming language.".to_string(),
        "FastEmbed provides fast embedding generation.".to_string(),
        "Machine learning models process natural language.".to_string(),
    ];

    let result = provider.embed_texts(&texts).await?;

    println!("   Generated {} embeddings", result.len());
    println!("   Embedding dimension: {}", result.dimension);

    for (i, (text, embedding)) in texts.iter().zip(result.embeddings.iter()).enumerate() {
        println!("   Text {}: \"{}\"", i + 1, text);
        println!(
            "   First 3 values: {:?}",
            &embedding[..3.min(embedding.len())]
        );
    }

    println!("\n🎉 Embedding generation completed successfully!");
    Ok(())
}
