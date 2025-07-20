//! Configuration for embedding models

use crate::error::{EmbedError, Result};
use derive_builder::Builder;
use std::path::{Path, PathBuf};

/// Configuration for embedding models
#[derive(Debug, Clone, Builder)]
#[builder(setter(into))]
pub struct EmbedConfig {
    /// Path to the base directory containing model files
    #[builder(default = r#"PathBuf::from("models")"#)]
    pub model_base_path: PathBuf,
    /// Name of the embedding model to use
    pub model_name: String,
    /// HuggingFace model repository (e.g., "answerdotai/ModernBERT-large")
    #[builder(default)]
    pub hf_model_repo: Option<String>,
    /// HuggingFace model revision/branch (e.g., "main")
    #[builder(default = r#"Some("main".to_string())"#)]
    pub hf_revision: Option<String>,
    /// Maximum batch size for embedding generation
    #[builder(default = "32")]
    pub batch_size: usize,
    /// Whether to normalize embeddings
    #[builder(default = "true")]
    pub normalize: bool,
}

impl EmbedConfig {
    /// Create a new embedding configuration using the builder
    pub fn builder() -> EmbedConfigBuilder {
        EmbedConfigBuilder::default()
    }

    /// Create a new embedding configuration (convenience method)
    pub fn new<P: AsRef<Path>>(model_base_path: P, model_name: impl Into<String>) -> Self {
        EmbedConfigBuilder::default()
            .model_base_path(model_base_path.as_ref())
            .model_name(model_name)
            .hf_model_repo(None::<String>)
            .build()
            .expect("Failed to build EmbedConfig")
    }

    /// Create a configuration for a HuggingFace model
    pub fn from_huggingface<P: AsRef<Path>>(
        model_base_path: P,
        model_name: impl Into<String>,
        hf_repo: impl Into<String>,
    ) -> Self {
        EmbedConfigBuilder::default()
            .model_base_path(model_base_path.as_ref())
            .model_name(model_name)
            .hf_model_repo(Some(hf_repo.into()))
            .batch_size(16usize) // Smaller batch for larger models
            .build()
            .expect("Failed to build EmbedConfig")
    }

    /// Create a default configuration for testing with a given path
    pub fn default_with_path<P: AsRef<Path>>(model_base_path: P) -> Self {
        Self::new(model_base_path, "snowflake-arctic-embed-xs")
    }

    /// Create a ModernBERT-large configuration
    pub fn modernbert_large<P: AsRef<Path>>(model_base_path: P) -> Self {
        Self::from_huggingface(
            model_base_path,
            "ModernBERT-large",
            "answerdotai/ModernBERT-large",
        )
    }

    /// Set the batch size for embedding generation (builder style)
    pub fn with_batch_size(self, batch_size: usize) -> Self {
        Self { batch_size, ..self }
    }

    /// Set whether to normalize embeddings (builder style)
    pub fn with_normalize(self, normalize: bool) -> Self {
        Self { normalize, ..self }
    }

    /// Set the HuggingFace revision (builder style)
    pub fn with_revision<S: Into<String>>(self, revision: S) -> Self {
        Self {
            hf_revision: Some(revision.into()),
            ..self
        }
    }

    /// Get the full path to the model directory
    pub fn model_path(&self) -> PathBuf {
        self.model_base_path.join(&self.model_name)
    }

    /// Get the path to the ONNX model file (ModernBERT uses model_q4.onnx)
    pub fn onnx_model_path(&self) -> PathBuf {
        let model_dir = self.model_path();

        // Check for ModernBERT naming convention first
        let modernbert_path = model_dir.join("onnx").join("model_q4.onnx");
        if modernbert_path.exists() {
            return modernbert_path;
        }

        // Fallback to standard naming
        model_dir.join("onnx").join("model_quantized.onnx")
    }

    /// Get the path to the tokenizer configuration
    pub fn tokenizer_path(&self) -> PathBuf {
        self.model_path().join("tokenizer.json")
    }

    /// Get the path to the config.json file
    pub fn config_path(&self) -> PathBuf {
        self.model_path().join("config.json")
    }

    /// Get the path to the special tokens map
    pub fn special_tokens_map_path(&self) -> PathBuf {
        self.model_path().join("special_tokens_map.json")
    }

    /// Check if this is a HuggingFace model
    pub fn is_huggingface_model(&self) -> bool {
        self.hf_model_repo.is_some()
    }

    /// Get the HuggingFace repository name
    pub fn hf_repo(&self) -> Option<&str> {
        self.hf_model_repo.as_deref()
    }

    /// Get the HuggingFace revision
    pub fn hf_revision(&self) -> &str {
        self.hf_revision.as_deref().unwrap_or("main")
    }

    /// Validate that all required model files exist
    pub fn validate(&self) -> Result<()> {
        let paths_to_check = [
            ("ONNX model", self.onnx_model_path()),
            ("tokenizer", self.tokenizer_path()),
            ("config", self.config_path()),
            ("special tokens map", self.special_tokens_map_path()),
        ];

        for (name, path) in &paths_to_check {
            if !path.exists() {
                tracing::error!("Missing {}: {}", name, path.display());
                return Err(EmbedError::ModelFileNotFound { path: path.clone() });
            }
        }

        tracing::debug!("Model validation successful for: {}", self.model_name);
        Ok(())
    }
}

impl Default for EmbedConfig {
    fn default() -> Self {
        EmbedConfigBuilder::default()
            .model_name("snowflake-arctic-embed-xs")
            .hf_model_repo(None::<String>)
            .build()
            .expect("Failed to build default EmbedConfig")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_config_creation() {
        let temp_dir = tempdir().unwrap();
        let config = EmbedConfig::new(temp_dir.path(), "test-model");

        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize);
        assert_eq!(config.model_path(), temp_dir.path().join("test-model"));
    }

    #[test]
    fn test_config_paths() {
        let temp_dir = tempdir().unwrap();
        let config = EmbedConfig::new(temp_dir.path(), "test-model");

        let expected_base = temp_dir.path().join("test-model");
        assert_eq!(
            config.onnx_model_path(),
            expected_base.join("onnx").join("model_quantized.onnx")
        );
        assert_eq!(
            config.tokenizer_path(),
            expected_base.join("tokenizer.json")
        );
        assert_eq!(config.config_path(), expected_base.join("config.json"));
        assert_eq!(
            config.special_tokens_map_path(),
            expected_base.join("special_tokens_map.json")
        );
    }

    #[test]
    fn test_config_builder_methods() {
        let temp_dir = tempdir().unwrap();
        let config = EmbedConfig::new(temp_dir.path(), "test-model")
            .with_batch_size(64)
            .with_normalize(false);

        assert_eq!(config.batch_size, 64);
        assert!(!config.normalize);
    }

    #[test]
    fn test_derive_builder_pattern() {
        let temp_dir = tempdir().unwrap();

        // Test using the builder directly
        let config = EmbedConfig::builder()
            .model_base_path(temp_dir.path())
            .model_name("custom-model")
            .batch_size(128usize)
            .normalize(false)
            .build()
            .unwrap();

        assert_eq!(config.model_name, "custom-model");
        assert_eq!(config.batch_size, 128);
        assert!(!config.normalize);
        assert_eq!(config.hf_revision, Some("main".to_string()));
    }

    #[test]
    fn test_builder_defaults() {
        // Test that defaults work correctly
        let config = EmbedConfig::builder()
            .model_name("test-model")
            .build()
            .unwrap();

        assert_eq!(config.model_base_path, PathBuf::from("models"));
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize);
        assert_eq!(config.hf_revision, Some("main".to_string()));
    }
}
