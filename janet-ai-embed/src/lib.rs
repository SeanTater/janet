//! # janet-ai-embed
//!
//! A library for generating embeddings using various providers, with a focus on
//! local ONNX models via fastembed. Designed for async operation and clean
//! abstraction to support multiple embedding providers.

pub mod config;
pub mod downloader;
pub mod error;
pub mod provider;

pub use config::{EmbedConfig, TokenizerConfig};
pub use downloader::ModelDownloader;
pub use error::{EmbedError, Result};
pub use provider::{EmbeddingProvider, EmbeddingResult, FastEmbedProvider};

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_config_creation() -> Result<()> {
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

        Ok(())
    }
}
