//! HuggingFace model downloading functionality

use crate::config::EmbedConfig;
use crate::error::{EmbedError, Result};
use hf_hub::api::tokio::Api;
use std::path::Path;
use tokio::fs;

/// Download a model from HuggingFace Hub if not already present locally.
///
/// This function checks if required model files exist locally, and if not,
/// downloads them from the specified HuggingFace repository.
///
/// If the configuration is not for a HuggingFace model, this function does nothing.
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
pub async fn download_model(config: &EmbedConfig) -> Result<()> {
    if !config.is_huggingface_model() {
        return Ok(());
    }

    let model_dir = config.model_path();

    // Check if already complete
    if is_model_complete(config) {
        tracing::info!(
            "Model {} already exists and is complete",
            config.model_name()
        );
        return Ok(());
    }

    let repo_id = config
        .hf_repo()
        .ok_or_else(|| EmbedError::invalid_config("HuggingFace repository not specified"))?;

    tracing::info!("Downloading model {} from {}", config.model_name(), repo_id);

    let api = Api::new().map_err(|e| EmbedError::External { source: e.into() })?;
    let repo = api.repo(hf_hub::Repo::model(repo_id.to_string()));

    // Create directories
    fs::create_dir_all(&model_dir).await?;
    fs::create_dir_all(model_dir.join("onnx")).await?;

    // Download files
    let downloads = [
        ("onnx/model_q4.onnx", model_dir.join("onnx/model_q4.onnx")),
        (
            "tokenizer.json",
            config.tokenizer_config().tokenizer_path.clone(),
        ),
        ("config.json", config.tokenizer_config().config_path.clone()),
        (
            "special_tokens_map.json",
            config.tokenizer_config().special_tokens_map_path.clone(),
        ),
    ];

    for (remote, local) in &downloads {
        if local.exists() {
            continue;
        }

        if let Some(parent) = local.parent() {
            fs::create_dir_all(parent).await?;
        }

        match repo.get(remote).await {
            Ok(file_path) => {
                fs::copy(&file_path, local).await?;
                tracing::debug!("Downloaded {}", remote);
            }
            Err(_e) if *remote == "special_tokens_map.json" => {
                create_fallback_special_tokens_map(local).await?;
            }
            Err(e) => return Err(EmbedError::External { source: e.into() }),
        }
    }

    // Optional tokenizer_config.json
    if let Some(path) = &config.tokenizer_config().tokenizer_config_path {
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await?;
            }
            if repo.get("tokenizer_config.json").await.is_ok() {
                // Download if available, but don't fail if missing
            }
        }
    }

    tracing::info!("Model {} downloaded successfully", config.model_name());
    Ok(())
}

/// Check if all required model files exist
fn is_model_complete(config: &EmbedConfig) -> bool {
    let onnx_file = config.model_path().join("onnx/model_q4.onnx");
    let tokenizer_config = config.tokenizer_config();

    [
        &onnx_file,
        &tokenizer_config.tokenizer_path,
        &tokenizer_config.config_path,
        &tokenizer_config.special_tokens_map_path,
    ]
    .iter()
    .all(|path| path.exists())
}

/// Create fallback special tokens map
async fn create_fallback_special_tokens_map(path: &Path) -> Result<()> {
    let fallback = serde_json::json!({
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]"
    });

    let json_str = serde_json::to_string_pretty(&fallback)
        .map_err(|e| EmbedError::External { source: e.into() })?;
    fs::write(path, json_str).await?;
    tracing::info!("Created fallback special_tokens_map.json");
    Ok(())
}

/// Legacy struct for backwards compatibility - prefer using `download_model()` function
#[deprecated(note = "Use `download_model()` function instead")]
pub struct ModelDownloader;

#[allow(deprecated)]
impl ModelDownloader {
    pub fn new() -> Self {
        Self
    }
    pub async fn ensure_model(&self, config: &EmbedConfig) -> Result<()> {
        download_model(config).await
    }
}

#[allow(deprecated)]
impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_is_model_complete_missing_files() {
        // Use a non-existent model name so files won't exist
        let config = EmbedConfig::new("nonexistent-test-model");

        assert!(!is_model_complete(&config)); // Should be false since no files exist
    }

    #[tokio::test]
    async fn test_download_model_skips_local() -> Result<()> {
        let config = EmbedConfig::new("test-model");

        // Should return Ok for non-HuggingFace models
        download_model(&config).await?;
        Ok(())
    }

    #[allow(deprecated)]
    #[tokio::test]
    async fn test_legacy_downloader() -> Result<()> {
        let config = EmbedConfig::new("test-model");

        let downloader = ModelDownloader::new();
        downloader.ensure_model(&config).await?;
        Ok(())
    }
}
