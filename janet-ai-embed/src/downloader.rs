//! HuggingFace model downloading functionality

use crate::config::EmbedConfig;
use crate::error::{EmbedError, Result};
use hf_hub::api::tokio::{Api, ApiRepo};
use std::path::Path;
use tokio::fs;

/// Downloads models from HuggingFace Hub
pub struct ModelDownloader {
    api: Api,
}

impl ModelDownloader {
    /// Create a new model downloader instance.
    ///
    /// This initializes the HuggingFace API client for downloading models.
    /// The downloader handles authentication automatically using environment
    /// variables or cached tokens.
    ///
    /// # Returns
    /// A new ModelDownloader ready to download models from HuggingFace Hub
    ///
    /// # Panics
    /// Panics if the HuggingFace API client cannot be initialized
    ///
    /// # Example
    /// ```
    /// use janet_ai_embed::ModelDownloader;
    ///
    /// let downloader = ModelDownloader::new();
    /// ```
    pub fn new() -> Self {
        Self {
            api: Api::new().expect("Failed to create HuggingFace API client"),
        }
    }

    /// Download a model from HuggingFace Hub if not already present locally.
    ///
    /// This method checks if the required model files exist locally, and if not,
    /// downloads them from the specified HuggingFace repository. It handles:
    /// - ONNX model files (model_q4.onnx or model_quantized.onnx)
    /// - Tokenizer files (tokenizer.json, config.json, etc.)
    /// - Progress tracking and error handling
    ///
    /// If the configuration is not for a HuggingFace model, this method does nothing.
    ///
    /// # Arguments
    /// * `config` - Configuration specifying the model to download
    ///
    /// # Returns
    /// `Ok(())` if the model is available (either downloaded or already present)
    ///
    /// # Errors
    /// - Network errors during download
    /// - File system errors when creating directories or writing files
    /// - HuggingFace API errors (repository not found, authentication, etc.)
    ///
    /// # Example
    /// ```no_run
    /// use janet_ai_embed::{ModelDownloader, EmbedConfig};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let downloader = ModelDownloader::new();
    /// let config = EmbedConfig::modernbert_large("/tmp/models");
    ///
    /// // Downloads ModernBERT-large if not already present
    /// downloader.ensure_model(&config).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn ensure_model(&self, config: &EmbedConfig) -> Result<()> {
        if !config.is_huggingface_model() {
            tracing::debug!("Not a HuggingFace model, skipping download");
            return Ok(());
        }

        let model_dir = config.model_path();

        // Check if model is already downloaded
        if self.is_model_complete(config).await? {
            tracing::info!("Model {} already exists and is complete", config.model_name);
            return Ok(());
        }

        let repo_id = config
            .hf_repo()
            .ok_or_else(|| EmbedError::invalid_config("HuggingFace repository not specified"))?;

        tracing::info!("Downloading model {} from {}", config.model_name, repo_id);

        // Create model directory
        fs::create_dir_all(&model_dir).await?;

        let repo = self.api.repo(hf_hub::Repo::model(repo_id.to_string()));

        // Download required files
        self.download_model_files(&repo, config).await?;

        tracing::info!("Model {} downloaded successfully", config.model_name);
        Ok(())
    }

    /// Check if the model is completely downloaded
    async fn is_model_complete(&self, config: &EmbedConfig) -> Result<bool> {
        let model_dir = config.model_path();
        let onnx_dir = model_dir.join("onnx");
        let tokenizer_config = &config.tokenizer_config;

        // Check for the specific ONNX file we download (model_q4.onnx)
        let onnx_file = onnx_dir.join("model_q4.onnx");

        let mut required_files = vec![
            onnx_file,
            tokenizer_config.tokenizer_path.clone(),
            tokenizer_config.config_path.clone(),
            tokenizer_config.special_tokens_map_path.clone(),
        ];

        // Add tokenizer_config.json if specified (it's optional)
        if let Some(ref path) = tokenizer_config.tokenizer_config_path {
            required_files.push(path.clone());
        }

        for file_path in &required_files {
            if !file_path.exists() {
                tracing::debug!("Missing file: {}", file_path.display());
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Download all required model files
    async fn download_model_files(&self, repo: &ApiRepo, config: &EmbedConfig) -> Result<()> {
        let model_dir = config.model_path();
        let onnx_dir = model_dir.join("onnx");
        let tokenizer_config = &config.tokenizer_config;

        // Create subdirectories
        fs::create_dir_all(&onnx_dir).await?;

        // Create parent directories for tokenizer files
        if let Some(parent) = tokenizer_config.tokenizer_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        if let Some(parent) = tokenizer_config.config_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        if let Some(parent) = tokenizer_config.special_tokens_map_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Files to download with their local paths
        let mut downloads = vec![
            ("onnx/model_q4.onnx", onnx_dir.join("model_q4.onnx")),
            ("tokenizer.json", tokenizer_config.tokenizer_path.clone()),
            ("config.json", tokenizer_config.config_path.clone()),
            (
                "special_tokens_map.json",
                tokenizer_config.special_tokens_map_path.clone(),
            ),
        ];

        // Add tokenizer_config.json if specified
        if let Some(ref path) = tokenizer_config.tokenizer_config_path {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await?;
            }
            downloads.push(("tokenizer_config.json", path.clone()));
        }

        for (remote_path, local_path) in &downloads {
            if local_path.exists() {
                tracing::debug!("File already exists: {}", local_path.display());
                continue;
            }

            tracing::info!("Downloading {} to {}", remote_path, local_path.display());

            match repo.get(remote_path).await {
                Ok(file_path) => {
                    // Copy the downloaded file to our model directory
                    fs::copy(&file_path, local_path)
                        .await
                        .map_err(|e| EmbedError::Io { source: e })?;
                    tracing::debug!("Successfully downloaded {}", remote_path);
                }
                Err(e) => {
                    tracing::warn!("Failed to download {}: {}", remote_path, e);
                    // For some files, we might want to continue anyway
                    if *remote_path == "special_tokens_map.json" {
                        // Create a minimal special tokens map if it doesn't exist
                        self.create_fallback_special_tokens_map(local_path).await?;
                    } else if *remote_path == "tokenizer_config.json" {
                        // tokenizer_config.json is optional - we handle this in the provider
                        tracing::info!(
                            "tokenizer_config.json not found, will generate minimal config"
                        );
                        continue;
                    } else {
                        return Err(EmbedError::External { source: e.into() });
                    }
                }
            }
        }

        Ok(())
    }

    /// Create a fallback special tokens map if the original is missing
    async fn create_fallback_special_tokens_map(&self, path: &Path) -> Result<()> {
        let fallback_content = serde_json::json!({
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]"
        });

        let content = serde_json::to_string_pretty(&fallback_content)
            .map_err(|e| EmbedError::External { source: e.into() })?;

        fs::write(path, content).await?;
        tracing::info!("Created fallback special_tokens_map.json");
        Ok(())
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_model_downloader_creation() {
        let _downloader = ModelDownloader::new();
        // Just test that we can create the downloader without panicking
    }

    #[tokio::test]
    async fn test_is_model_complete_missing_files() -> Result<()> {
        let temp_dir = tempdir().unwrap();
        let config = EmbedConfig::modernbert_large(temp_dir.path());
        let downloader = ModelDownloader::new();

        let is_complete = downloader.is_model_complete(&config).await?;
        assert!(!is_complete); // Should be false since no files exist

        Ok(())
    }
}
