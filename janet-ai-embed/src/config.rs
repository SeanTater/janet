//! Configuration for embedding models

use crate::error::{EmbedError, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Configuration for tokenizer files
#[derive(Debug, Clone, Serialize)]
pub struct TokenizerConfig {
    /// Path to the tokenizer.json file
    pub tokenizer_path: PathBuf,
    /// Path to the config.json file
    pub config_path: PathBuf,
    /// Path to the special_tokens_map.json file
    pub special_tokens_map_path: PathBuf,
    /// Path to the tokenizer_config.json file (optional, will generate minimal if missing)
    pub tokenizer_config_path: Option<PathBuf>,
}

impl TokenizerConfig {
    /// Create a new tokenizer configuration with all required paths.
    ///
    /// # Arguments
    /// * `tokenizer_path` - Path to the tokenizer.json file
    /// * `config_path` - Path to the config.json file
    /// * `special_tokens_map_path` - Path to the special_tokens_map.json file
    /// * `tokenizer_config_path` - Optional path to tokenizer_config.json file
    ///
    /// # Returns
    /// A new TokenizerConfig instance
    pub fn new<P: AsRef<Path>>(
        tokenizer_path: P,
        config_path: P,
        special_tokens_map_path: P,
        tokenizer_config_path: Option<P>,
    ) -> Self {
        Self {
            tokenizer_path: tokenizer_path.as_ref().to_path_buf(),
            config_path: config_path.as_ref().to_path_buf(),
            special_tokens_map_path: special_tokens_map_path.as_ref().to_path_buf(),
            tokenizer_config_path: tokenizer_config_path.map(|p| p.as_ref().to_path_buf()),
        }
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
        Self {
            tokenizer_path: model_dir.join("tokenizer.json"),
            config_path: model_dir.join("config.json"),
            special_tokens_map_path: model_dir.join("special_tokens_map.json"),
            tokenizer_config_path: Some(model_dir.join("tokenizer_config.json")),
        }
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
        Self {
            tokenizer_path: tokenizer_path.as_ref().to_path_buf(),
            config_path: config_path.as_ref().to_path_buf(),
            special_tokens_map_path: special_tokens_map_path.as_ref().to_path_buf(),
            tokenizer_config_path: None,
        }
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
#[derive(Debug, Clone, Serialize)]
pub enum EmbedConfig {
    /// Local model configuration
    Local {
        /// Name of the embedding model to use
        model_name: String,
    },
    /// HuggingFace model configuration
    HuggingFace {
        /// Name of the embedding model to use
        model_name: String,
    },
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self::modernbert_large()
    }
}

impl EmbedConfig {
    /// Get the standard model base path ($HOME/.janet/models)
    fn model_base_path() -> PathBuf {
        if let Some(home) = std::env::var_os("HOME") {
            PathBuf::from(home).join(".janet").join("models")
        } else {
            PathBuf::from(".janet").join("models")
        }
    }

    /// Creates basic local model configuration. See module docs for details.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self::Local {
            model_name: model_name.into(),
        }
    }

    /// Creates configuration for HuggingFace model download. See module docs for details.
    pub fn from_huggingface(model_name: impl Into<String>) -> Self {
        Self::HuggingFace {
            model_name: model_name.into(),
        }
    }

    /// Creates configuration for ModernBERT-large model from HuggingFace. See module docs for details.
    pub fn modernbert_large() -> Self {
        Self::from_huggingface("ModernBERT-large")
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
        Self::model_base_path().join(self.model_name())
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

    /// Get the tokenizer configuration (auto-generated from model path)
    pub fn tokenizer_config(&self) -> TokenizerConfig {
        TokenizerConfig::standard(self.model_path())
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        match self {
            Self::Local { model_name } | Self::HuggingFace { model_name } => model_name,
        }
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
        matches!(self, Self::HuggingFace { .. })
    }

    /// Get the HuggingFace repository name (inferred from model name)
    pub fn hf_repo(&self) -> Option<&str> {
        match self {
            Self::Local { .. } => None,
            Self::HuggingFace { model_name } => {
                // Map common model names to their repos
                match model_name.as_str() {
                    "ModernBERT-large" => Some("answerdotai/ModernBERT-large"),
                    _ => Some("unknown/repo"), // Fallback
                }
            }
        }
    }

    /// Get the HuggingFace revision (always "main")
    pub fn hf_revision(&self) -> &str {
        "main"
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
        self.tokenizer_config().validate()?;

        tracing::debug!("Model validation successful for: {}", self.model_name());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_config_creation() {
        let config = EmbedConfig::new("test-model");

        assert_eq!(config.model_name(), "test-model");
        assert_eq!(
            config.model_path(),
            EmbedConfig::model_base_path().join("test-model")
        );
        assert!(!config.is_huggingface_model());
    }

    #[test]
    fn test_config_paths() {
        let config = EmbedConfig::new("test-model");

        let expected_base = EmbedConfig::model_base_path().join("test-model");
        assert_eq!(
            config.onnx_model_path(),
            expected_base.join("onnx").join("model_quantized.onnx")
        );
        assert_eq!(
            config.tokenizer_config().tokenizer_path,
            expected_base.join("tokenizer.json")
        );
        assert_eq!(
            config.tokenizer_config().config_path,
            expected_base.join("config.json")
        );
        assert_eq!(
            config.tokenizer_config().special_tokens_map_path,
            expected_base.join("special_tokens_map.json")
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = EmbedConfig::new("test-model");

        // All values are now static/inferred
        assert_eq!(config.model_name(), "test-model");
        assert_eq!(config.hf_revision(), "main");
        assert!(!config.is_huggingface_model());

        // Test default is ModernBERT
        let default_config = EmbedConfig::default();
        assert_eq!(default_config.model_name(), "ModernBERT-large");
        assert!(default_config.is_huggingface_model());
    }

    #[test]
    fn test_direct_enum_construction() {
        // Test using direct enum construction for local model
        let local_config = EmbedConfig::Local {
            model_name: "custom-model".to_string(),
        };

        assert_eq!(local_config.model_name(), "custom-model");
        assert!(!local_config.is_huggingface_model());

        // Test using direct enum construction for HuggingFace model
        let hf_config = EmbedConfig::HuggingFace {
            model_name: "ModernBERT-large".to_string(),
        };

        assert_eq!(hf_config.model_name(), "ModernBERT-large");
        assert!(hf_config.is_huggingface_model());
        assert_eq!(hf_config.hf_repo(), Some("answerdotai/ModernBERT-large"));
    }

    #[test]
    fn test_constructor_defaults() {
        // Test that new() method sets correct defaults
        let config = EmbedConfig::new("test-model");

        assert_eq!(
            config.model_path(),
            EmbedConfig::model_base_path().join("test-model")
        );
        assert_eq!(config.hf_revision(), "main");
        assert!(!config.is_huggingface_model());
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
