//! ModernBERT-large embedding example using real HuggingFace model

use janet_ai_embed::{EmbedConfig, EmbeddingProvider, FastEmbedProvider, Result};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🤖 Janet AI Embed - ModernBERT-large Example");
    println!("============================================");

    // Get the model cache directory from environment or use default
    let cache_dir = env::var("JANET_MODEL_CACHE")
        .unwrap_or_else(|_| "models".to_string());

    println!("📁 Using model cache directory: {}", cache_dir);

    // Create configuration for ModernBERT-large
    let config = EmbedConfig::modernbert_large(&cache_dir)
        .with_batch_size(4)
        .with_normalize(true);

    println!("📝 Creating FastEmbed provider with ModernBERT-large:");
    println!("   Model: {}", config.model_name);
    println!("   HF Repo: {}", config.hf_repo().unwrap_or("N/A"));
    println!("   Batch size: {}", config.batch_size);
    println!("   Normalize: {}", config.normalize);

    // Create and initialize the provider (this will download the model if needed)
    println!("\n⬇️  Downloading and initializing ModernBERT-large model...");
    println!("   (This may take several minutes for the first run)");
    
    let provider = match FastEmbedProvider::create(config).await {
        Ok(provider) => {
            println!("✅ ModernBERT-large provider initialized successfully!");
            println!("   Dimension: {}", provider.embedding_dimension());
            println!("   Provider: {}", provider.provider_name());
            provider
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize ModernBERT-large: {}", e);
            eprintln!("   This might be due to:");
            eprintln!("   - Network connectivity issues");
            eprintln!("   - Missing model files in HuggingFace repository");
            eprintln!("   - Insufficient disk space");
            eprintln!("   - ONNX runtime compatibility issues");
            return Err(e);
        }
    };

    // Test with single text
    println!("\n📊 Testing ModernBERT-large with single text...");
    let test_text = "ModernBERT is a state-of-the-art encoder-only language model designed for improved efficiency and performance.";
    
    let embedding = provider.embed_text(test_text).await?;
    println!("   Text: \"{}\"", test_text);
    println!("   Embedding dimension: {}", embedding.len());
    println!("   First 5 values: {:?}", &embedding[..5.min(embedding.len())]);

    // Test with multiple texts
    println!("\n📊 Testing ModernBERT-large with multiple texts...");
    let texts = vec![
        "ModernBERT provides efficient transformer-based embeddings.".to_string(),
        "The model uses advanced architectural improvements.".to_string(),
        "Quantized ONNX models enable fast inference.".to_string(),
        "Semantic embeddings capture meaning and context.".to_string(),
    ];

    let result = provider.embed_texts(&texts).await?;
    
    println!("   Generated {} embeddings", result.len());
    println!("   Embedding dimension: {}", result.dimension);
    
    for (i, (text, embedding)) in texts.iter().zip(result.embeddings.iter()).enumerate() {
        let display_text = if text.len() > 50 { 
            format!("{}...", &text[..47]) 
        } else { 
            text.clone() 
        };
        println!("   Text {}: \"{}\"", i + 1, display_text);
        println!("            First 3 values: {:?}", &embedding[..3.min(embedding.len())]);
    }

    // Calculate similarity between first two embeddings
    if result.embeddings.len() >= 2 {
        let emb1 = &result.embeddings[0];
        let emb2 = &result.embeddings[1];
        
        // Simple cosine similarity calculation
        let dot_product: f32 = emb1.iter().zip(emb2.iter())
            .map(|(a, b)| a.to_f32() * b.to_f32())
            .sum();
        let norm1: f32 = emb1.iter().map(|x| x.to_f32().powi(2)).sum::<f32>().sqrt();
        let norm2: f32 = emb2.iter().map(|x| x.to_f32().powi(2)).sum::<f32>().sqrt();
        let similarity = dot_product / (norm1 * norm2);
        
        println!("\n🔍 Similarity between first two texts: {:.4}", similarity);
    }

    println!("\n🎉 ModernBERT-large embedding generation completed successfully!");
    println!("💡 Model files are cached at: {}", cache_dir);
    
    Ok(())
}