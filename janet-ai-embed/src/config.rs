//! Configuration for embedding models

use crate::error::{EmbedError, Result};
use derive_builder::Builder;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Configuration for tokenizer files
#[derive(Debug, Clone, Builder, Serialize)]
#[builder(setter(into))]
pub struct TokenizerConfig {
    /// Path to the tokenizer.json file
    pub tokenizer_path: PathBuf,
    /// Path to the config.json file
    pub config_path: PathBuf,
    /// Path to the special_tokens_map.json file
    pub special_tokens_map_path: PathBuf,
    /// Path to the tokenizer_config.json file (optional, will generate minimal if missing)
    #[builder(default)]
    pub tokenizer_config_path: Option<PathBuf>,
}

impl TokenizerConfig {
    /// Create a new tokenizer configuration using the builder pattern.
    ///
    /// This returns a builder that allows you to set each field individually.
    /// Use this when you need fine-grained control over tokenizer file paths.
    ///
    /// # Returns
    /// A builder instance for constructing TokenizerConfig
    ///
    /// # Example
    pub fn builder() -> TokenizerConfigBuilder {
        TokenizerConfigBuilder::default()
    }

    /// Create a standard tokenizer configuration for a model directory.
    ///
    /// This assumes the standard HuggingFace model layout with files:
    /// - `tokenizer.json` - The tokenizer configuration
    /// - `config.json` - Model configuration
    /// - `special_tokens_map.json` - Special token mappings
    /// - `tokenizer_config.json` - Additional tokenizer configuration (optional)
    ///
    /// # Arguments
    /// * `model_dir` - Path to the directory containing the tokenizer files
    ///
    /// # Returns
    /// A configured TokenizerConfig with standard file paths
    ///
    /// # Example
    pub fn standard<P: AsRef<Path>>(model_dir: P) -> Self {
        let model_dir = model_dir.as_ref();
        TokenizerConfigBuilder::default()
            .tokenizer_path(model_dir.join("tokenizer.json"))
            .config_path(model_dir.join("config.json"))
            .special_tokens_map_path(model_dir.join("special_tokens_map.json"))
            .tokenizer_config_path(Some(model_dir.join("tokenizer_config.json")))
            .build()
            .expect("Failed to build TokenizerConfig")
    }

    /// Create a tokenizer configuration with custom file paths.
    ///
    /// Use this when your tokenizer files are in non-standard locations or have
    /// different names than the HuggingFace defaults.
    ///
    /// # Arguments
    /// * `tokenizer_path` - Path to the tokenizer.json file
    /// * `config_path` - Path to the config.json file
    /// * `special_tokens_map_path` - Path to the special_tokens_map.json file
    ///
    /// # Returns
    /// A configured TokenizerConfig with the specified file paths
    ///
    /// # Example
    pub fn custom<P1, P2, P3>(
        tokenizer_path: P1,
        config_path: P2,
        special_tokens_map_path: P3,
    ) -> Self
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        P3: AsRef<Path>,
    {
        TokenizerConfigBuilder::default()
            .tokenizer_path(tokenizer_path.as_ref())
            .config_path(config_path.as_ref())
            .special_tokens_map_path(special_tokens_map_path.as_ref())
            .build()
            .expect("Failed to build TokenizerConfig")
    }

    /// Validate that all required tokenizer files exist on the filesystem.
    ///
    /// This checks that the tokenizer.json, config.json, and special_tokens_map.json
    /// files exist and are readable. The tokenizer_config.json file is optional.
    ///
    /// # Returns
    /// `Ok(())` if all required files exist, or an error describing the missing file
    ///
    /// # Errors
    /// Returns `EmbedError::InvalidConfig` if any required file is missing
    ///
    /// # Example
    pub fn validate(&self) -> Result<()> {
        let paths_to_check = [
            ("tokenizer", &self.tokenizer_path),
            ("config", &self.config_path),
            ("special tokens map", &self.special_tokens_map_path),
        ];

        for (name, path) in &paths_to_check {
            if !path.exists() {
                tracing::error!("Missing {}: {}", name, path.display());
                return Err(EmbedError::ModelFileNotFound {
                    path: path.to_path_buf(),
                });
            }
        }

        // tokenizer_config.json is optional
        if let Some(path) = &self.tokenizer_config_path {
            if !path.exists() {
                tracing::warn!(
                    "tokenizer_config.json not found at {}, will generate minimal config",
                    path.display()
                );
            }
        }

        tracing::debug!("Tokenizer validation successful");
        Ok(())
    }
}

/// Configuration for embedding models
#[derive(Debug, Clone, Builder, Serialize)]
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
    /// Tokenizer configuration
    pub tokenizer_config: TokenizerConfig,
}

impl EmbedConfig {
    /// Creates a new builder for custom configuration. See module docs for usage examples.
    pub fn builder() -> EmbedConfigBuilder {
        EmbedConfigBuilder::default()
    }

    /// Creates basic configuration with required parameters. See module docs for details.
    pub fn new<P: AsRef<Path>>(
        model_base_path: P,
        model_name: impl Into<String>,
        tokenizer_config: TokenizerConfig,
    ) -> Self {
        EmbedConfigBuilder::default()
            .model_base_path(model_base_path.as_ref())
            .model_name(model_name)
            .hf_model_repo(None::<String>)
            .tokenizer_config(tokenizer_config)
            .build()
            .expect("Failed to build EmbedConfig")
    }

    /// Creates configuration for HuggingFace model download. See module docs for details.
    pub fn from_huggingface<P: AsRef<Path>>(
        model_base_path: P,
        model_name: impl Into<String>,
        hf_repo: impl Into<String>,
        tokenizer_config: TokenizerConfig,
    ) -> Self {
        EmbedConfigBuilder::default()
            .model_base_path(model_base_path.as_ref())
            .model_name(model_name)
            .hf_model_repo(Some(hf_repo.into()))
            .batch_size(16usize) // Smaller batch for larger models
            .tokenizer_config(tokenizer_config)
            .build()
            .expect("Failed to build EmbedConfig")
    }

    /// Creates default configuration with Arctic Embed XS model. See module docs for details.
    pub fn default_with_path<P: AsRef<Path>>(model_base_path: P) -> Self {
        let model_dir = model_base_path.as_ref().join("snowflake-arctic-embed-xs");
        let tokenizer_config = TokenizerConfig::standard(&model_dir);
        Self::new(
            model_base_path,
            "snowflake-arctic-embed-xs",
            tokenizer_config,
        )
    }

    /// Creates configuration for ModernBERT-large model from HuggingFace. See module docs for details.
    pub fn modernbert_large<P: AsRef<Path>>(model_base_path: P) -> Self {
        let model_dir = model_base_path.as_ref().join("ModernBERT-large");
        let tokenizer_config = TokenizerConfig::standard(&model_dir);
        Self::from_huggingface(
            model_base_path,
            "ModernBERT-large",
            "answerdotai/ModernBERT-large",
            tokenizer_config,
        )
    }

    /// Sets batch size for embedding generation. See module docs for usage examples.
    pub fn with_batch_size(self, batch_size: usize) -> Self {
        Self { batch_size, ..self }
    }

    /// Sets embedding normalization. See module docs for usage examples.
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

    /// Get the full path where this model's files are stored.
    ///
    /// This combines the base path with the model name to create the directory
    /// where all model files (ONNX, tokenizer, etc.) are located.
    ///
    /// # Returns
    /// The complete path to the model directory
    ///
    /// # Example
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

    /// Get the tokenizer configuration
    pub fn tokenizer_config(&self) -> &TokenizerConfig {
        &self.tokenizer_config
    }

    /// Check if this configuration is set up for a HuggingFace model.
    ///
    /// Returns `true` if this config has a HuggingFace repository specified,
    /// meaning the model will be downloaded from HuggingFace Hub.
    ///
    /// # Returns
    /// `true` if configured for HuggingFace download, `false` for local models
    ///
    /// # Example
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
        // Validate ONNX model
        let onnx_path = self.onnx_model_path();
        if !onnx_path.exists() {
            tracing::error!("Missing ONNX model: {}", onnx_path.display());
            return Err(EmbedError::ModelFileNotFound { path: onnx_path });
        }

        // Validate tokenizer configuration
        self.tokenizer_config.validate()?;

        tracing::debug!("Model validation successful for: {}", self.model_name);
        Ok(())
    }
}

impl Default for EmbedConfig {
    fn default() -> Self {
        let model_dir = PathBuf::from("models").join("snowflake-arctic-embed-xs");
        let tokenizer_config = TokenizerConfig::standard(&model_dir);
        EmbedConfigBuilder::default()
            .model_name("snowflake-arctic-embed-xs")
            .hf_model_repo(None::<String>)
            .tokenizer_config(tokenizer_config)
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
        let model_dir = temp_dir.path().join("test-model");
        let tokenizer_config = TokenizerConfig::standard(&model_dir);
        let config = EmbedConfig::new(temp_dir.path(), "test-model", tokenizer_config);

        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize);
        assert_eq!(config.model_path(), temp_dir.path().join("test-model"));
    }

    #[test]
    fn test_config_paths() {
        let temp_dir = tempdir().unwrap();
        let model_dir = temp_dir.path().join("test-model");
        let tokenizer_config = TokenizerConfig::standard(&model_dir);
        let config = EmbedConfig::new(temp_dir.path(), "test-model", tokenizer_config);

        let expected_base = temp_dir.path().join("test-model");
        assert_eq!(
            config.onnx_model_path(),
            expected_base.join("onnx").join("model_quantized.onnx")
        );
        assert_eq!(
            config.tokenizer_config.tokenizer_path,
            expected_base.join("tokenizer.json")
        );
        assert_eq!(
            config.tokenizer_config.config_path,
            expected_base.join("config.json")
        );
        assert_eq!(
            config.tokenizer_config.special_tokens_map_path,
            expected_base.join("special_tokens_map.json")
        );
    }

    #[test]
    fn test_config_builder_methods() {
        let temp_dir = tempdir().unwrap();
        let model_dir = temp_dir.path().join("test-model");
        let tokenizer_config = TokenizerConfig::standard(&model_dir);
        let config = EmbedConfig::new(temp_dir.path(), "test-model", tokenizer_config)
            .with_batch_size(64)
            .with_normalize(false);

        assert_eq!(config.batch_size, 64);
        assert!(!config.normalize);
    }

    #[test]
    fn test_derive_builder_pattern() {
        let temp_dir = tempdir().unwrap();
        let model_dir = temp_dir.path().join("custom-model");
        let tokenizer_config = TokenizerConfig::standard(&model_dir);

        // Test using the builder directly
        let config = EmbedConfig::builder()
            .model_base_path(temp_dir.path())
            .model_name("custom-model")
            .batch_size(128usize)
            .normalize(false)
            .tokenizer_config(tokenizer_config)
            .build()
            .unwrap();

        assert_eq!(config.model_name, "custom-model");
        assert_eq!(config.batch_size, 128);
        assert!(!config.normalize);
        assert_eq!(config.hf_revision, Some("main".to_string()));
    }

    #[test]
    fn test_builder_defaults() {
        let model_dir = PathBuf::from("models").join("test-model");
        let tokenizer_config = TokenizerConfig::standard(&model_dir);

        // Test that defaults work correctly
        let config = EmbedConfig::builder()
            .model_name("test-model")
            .tokenizer_config(tokenizer_config)
            .build()
            .unwrap();

        assert_eq!(config.model_base_path, PathBuf::from("models"));
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize);
        assert_eq!(config.hf_revision, Some("main".to_string()));
    }

    #[test]
    fn test_tokenizer_config() {
        let temp_dir = tempdir().unwrap();
        let model_dir = temp_dir.path().join("test-model");

        // Test standard tokenizer config
        let tokenizer_config = TokenizerConfig::standard(&model_dir);
        assert_eq!(
            tokenizer_config.tokenizer_path,
            model_dir.join("tokenizer.json")
        );
        assert_eq!(tokenizer_config.config_path, model_dir.join("config.json"));
        assert_eq!(
            tokenizer_config.special_tokens_map_path,
            model_dir.join("special_tokens_map.json")
        );
        assert_eq!(
            tokenizer_config.tokenizer_config_path,
            Some(model_dir.join("tokenizer_config.json"))
        );

        // Test custom tokenizer config
        let custom_config = TokenizerConfig::custom(
            temp_dir.path().join("custom_tokenizer.json"),
            temp_dir.path().join("custom_config.json"),
            temp_dir.path().join("custom_special_tokens.json"),
        );
        assert_eq!(
            custom_config.tokenizer_path,
            temp_dir.path().join("custom_tokenizer.json")
        );
        assert_eq!(
            custom_config.config_path,
            temp_dir.path().join("custom_config.json")
        );
        assert_eq!(
            custom_config.special_tokens_map_path,
            temp_dir.path().join("custom_special_tokens.json")
        );
        assert_eq!(custom_config.tokenizer_config_path, None);
    }
}
