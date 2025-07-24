//! # janet-ai-embed
//!
//! A high-performance library for generating text embeddings using various providers,
//! with a focus on local ONNX models via FastEmbed. Designed for async operation with
//! clean abstractions to support multiple embedding providers.
//!
//! ## Features
//!
//! - **Local ONNX Models**: Run embedding models locally without external API calls
//! - **Async-First Design**: Full async/await support with tokio integration
//! - **Model Caching**: Intelligent caching to avoid reloading models
//! - **HuggingFace Integration**: Automatic model downloading from HuggingFace Hub
//! - **Half-Precision**: Memory-efficient f16 embeddings for better performance
//! - **Configurable**: Flexible configuration with sensible defaults
//!
//! ## Quick Start
//!
//! ```no_run
//! use janet_ai_embed::{FastEmbedProvider, EmbedConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a provider with ModernBERT-large model
//! let provider = FastEmbedProvider::create(
//!     EmbedConfig::modernbert_large("/tmp/models")
//! ).await?;
//!
//! // Generate embeddings for text
//! let texts = vec!["Hello world".to_string(), "How are you?".to_string()];
//! let result = provider.embed_texts(&texts).await?;
//!
//! println!("Generated {} embeddings of dimension {}",
//!          result.len(), result.dimension);
//! # Ok(())
//! # }
//! ```
//!
//! ## Supported Models
//!
//! The library supports various embedding models including:
//!
//! - **ModernBERT**: High-quality general-purpose embeddings
//! - **Snowflake Arctic Embed**: Lightweight and efficient
//! - **Custom ONNX Models**: Load your own fine-tuned models
//!
//! ## Architecture
//!
//! The crate is organized into several modules:
//!
//! - [`config`]: Configuration types for models and tokenizers
//! - [`provider`]: Embedding provider implementations and traits
//! - [`downloader`]: HuggingFace model downloading functionality
//! - [`error`]: Error types and result handling
//!
//! ## Memory Usage
//!
//! The library uses half-precision (f16) embeddings by default to reduce memory
//! usage while maintaining accuracy. Models are cached globally to avoid
//! redundant loading when multiple providers use the same configuration.
//!
//! ## Error Handling
//!
//! All operations return [`Result<T>`] using the crate's [`EmbedError`] type,
//! which provides detailed context about failures including configuration errors,
//! model loading issues, and runtime failures.

pub mod config;
pub mod downloader;
pub mod error;
pub mod provider;

// Re-export main types for easy access
pub use config::{EmbedConfig, TokenizerConfig};
pub use downloader::ModelDownloader;
pub use error::{EmbedError, Result};
pub use provider::{EmbeddingProvider, EmbeddingResult, FastEmbedProvider};

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_config_creation() {
        // Test basic configuration creation without actually downloading models
        let temp_dir = tempdir().unwrap();
        let config = EmbedConfig::default_with_path(temp_dir.path());

        assert_eq!(config.model_name, "snowflake-arctic-embed-xs");
        assert!(!config.is_huggingface_model());

        // Test ModernBERT config
        let modernbert_config = EmbedConfig::modernbert_large(temp_dir.path());
        assert_eq!(modernbert_config.model_name, "ModernBERT-large");
        assert!(modernbert_config.is_huggingface_model());
        assert_eq!(
            modernbert_config.hf_repo(),
            Some("answerdotai/ModernBERT-large")
        );
    }
}
