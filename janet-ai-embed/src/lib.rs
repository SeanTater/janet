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
#[allow(deprecated)]
pub use downloader::{ModelDownloader, download_model};
pub use error::{EmbedError, Result};
pub use provider::{EmbeddingProvider, EmbeddingResult, FastEmbedProvider};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        // Test basic configuration creation without actually downloading models
        let config = EmbedConfig::default();

        assert_eq!(config.model_name(), "ModernBERT-large");
        assert!(config.is_huggingface_model());

        // Test ModernBERT config
        let modernbert_config = EmbedConfig::modernbert_large();
        assert_eq!(modernbert_config.model_name(), "ModernBERT-large");
        assert!(modernbert_config.is_huggingface_model());
        assert_eq!(
            modernbert_config.hf_repo(),
            Some("answerdotai/ModernBERT-large")
        );
    }
}
