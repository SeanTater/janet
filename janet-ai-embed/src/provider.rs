//! Embedding provider implementations

use crate::config::EmbedConfig;
use crate::downloader::ModelDownloader;
use crate::error::{EmbedError, Result};
use async_trait::async_trait;
use fastembed::{
    EmbeddingModel, InitOptions, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use half::f16;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::fs;

/// Result of embedding generation
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The generated embeddings, one per input text
    pub embeddings: Vec<Vec<f16>>,
    /// The dimension of each embedding vector
    pub dimension: usize,
}

impl EmbeddingResult {
    /// Create a new embedding result
    pub fn new(embeddings: Vec<Vec<f16>>) -> Self {
        let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);
        Self {
            embeddings,
            dimension,
        }
    }

    /// Get the number of embeddings
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if the result is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

/// Type alias for cached model entries (model, dimension)
type ModelCacheEntry = (Arc<Mutex<TextEmbedding>>, usize);

/// Global cache for initialized embedding models to avoid reloading
static MODEL_CACHE: OnceLock<Mutex<HashMap<String, ModelCacheEntry>>> = OnceLock::new();

/// Get the global model cache
fn get_model_cache() -> &'static Mutex<HashMap<String, ModelCacheEntry>> {
    MODEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Trait for embedding providers that can generate embeddings from text
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for a single text
    async fn embed_text(&self, text: &str) -> Result<Vec<f16>>;

    /// Generate embeddings for multiple texts (batch processing)
    async fn embed_texts(&self, texts: &[String]) -> Result<EmbeddingResult>;

    /// Get the dimension of embeddings produced by this provider
    fn embedding_dimension(&self) -> usize;

    /// Get the name/identifier of this provider
    fn provider_name(&self) -> &str;
}

/// FastEmbed-based embedding provider using real ONNX models
pub struct FastEmbedProvider {
    config: EmbedConfig,
    model: Option<Arc<Mutex<TextEmbedding>>>,
    dimension: usize,
}

impl FastEmbedProvider {
    /// Create a new FastEmbed provider with the given configuration
    pub fn new(config: EmbedConfig) -> Self {
        Self {
            config,
            model: None,
            dimension: 1024, // Default dimension for ModernBERT-large
        }
    }

    /// Initialize the provider by downloading and loading the model
    pub async fn initialize(&mut self) -> Result<()> {
        tracing::info!(
            "Initializing FastEmbed provider for model: {}",
            self.config.model_name
        );

        // Create a cache key based on the model configuration
        let cache_key = self.create_cache_key();

        // Check if model is already cached
        let cached_data = {
            let cache = get_model_cache().lock().unwrap();
            cache
                .get(&cache_key)
                .map(|(model, dim)| (Arc::clone(model), *dim))
        };

        if let Some((cached_model, cached_dimension)) = cached_data {
            tracing::info!("Using cached model for: {}", self.config.model_name);
            self.model = Some(cached_model);
            self.dimension = cached_dimension;
            return self.validate_model().await;
        }

        // Check if this is a HuggingFace model that needs downloading
        if self.config.is_huggingface_model() {
            tracing::info!("Downloading HuggingFace model: {}", self.config.model_name);
            let downloader = ModelDownloader::new();
            downloader.ensure_model(&self.config).await?;

            // Try to load the user-defined model
            let (model, dimension) = self.load_user_defined_model().await?;
            let model_arc = Arc::new(Mutex::new(model));

            // Cache the model
            {
                let mut cache = get_model_cache().lock().unwrap();
                cache.insert(cache_key, (Arc::clone(&model_arc), dimension));
            }

            self.model = Some(model_arc);
            self.dimension = dimension;
        } else {
            tracing::info!("Using built-in fastembed model: {}", self.config.model_name);

            // Load model in a blocking task
            let config = self.config.clone();
            let (model, dimension) =
                tokio::task::spawn_blocking(move || -> Result<(TextEmbedding, usize)> {
                    tracing::info!("Loading embedding model: {}", config.model_name);

                    let init_options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                        .with_show_download_progress(true);

                    let mut model = TextEmbedding::try_new(init_options)
                        .map_err(|e| EmbedError::External { source: e })?;

                    // Get dimension by generating a test embedding
                    let test_embeddings = model
                        .embed(vec!["test".to_string()], None)
                        .map_err(|e| EmbedError::External { source: e })?;
                    let dimension = test_embeddings.first().map(|emb| emb.len()).unwrap_or(384);

                    tracing::info!("Model loaded successfully. Dimension: {}", dimension);
                    Ok((model, dimension))
                })
                .await??;

            let model_arc = Arc::new(Mutex::new(model));

            // Cache the model
            {
                let mut cache = get_model_cache().lock().unwrap();
                cache.insert(cache_key, (Arc::clone(&model_arc), dimension));
            }

            self.model = Some(model_arc);
            self.dimension = dimension;
        }

        // Validate the model works correctly
        self.validate_model().await
    }

    /// Create and initialize a new FastEmbed provider
    pub async fn create(config: EmbedConfig) -> Result<Self> {
        let mut provider = Self::new(config);
        provider.initialize().await?;
        Ok(provider)
    }

    /// Create a cache key based on the model configuration
    fn create_cache_key(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.config.model_name,
            self.config.batch_size,
            self.config.normalize,
            self.config.hf_revision()
        )
    }

    /// Load a user-defined ONNX model from downloaded HuggingFace files
    async fn load_user_defined_model(&self) -> Result<(TextEmbedding, usize)> {
        tracing::info!("Loading user-defined model: {}", self.config.model_name);

        // Read all required files
        let onnx_file = fs::read(self.config.onnx_model_path())
            .await
            .map_err(|e| EmbedError::Io { source: e })?;

        let tokenizer_file = fs::read(self.config.tokenizer_path())
            .await
            .map_err(|e| EmbedError::Io { source: e })?;

        let config_file = fs::read(self.config.config_path())
            .await
            .map_err(|e| EmbedError::Io { source: e })?;

        let special_tokens_map_file = fs::read(self.config.special_tokens_map_path())
            .await
            .map_err(|e| EmbedError::Io { source: e })?;

        // Check if tokenizer_config.json exists, create a minimal one if not
        let tokenizer_config_path = self.config.model_path().join("tokenizer_config.json");
        let tokenizer_config_file = if tokenizer_config_path.exists() {
            fs::read(&tokenizer_config_path)
                .await
                .map_err(|e| EmbedError::Io { source: e })?
        } else {
            // Create a minimal tokenizer config
            let minimal_config = serde_json::json!({
                "clean_up_tokenization_spaces": true,
                "do_lower_case": false,
                "model_max_length": 512,
                "tokenizer_class": "BertTokenizer"
            });
            serde_json::to_vec_pretty(&minimal_config)
                .map_err(|e| EmbedError::External { source: e.into() })?
        };

        // Create the tokenizer files struct
        let tokenizer_files = TokenizerFiles {
            tokenizer_file,
            config_file,
            special_tokens_map_file,
            tokenizer_config_file,
        };

        // Create the user-defined embedding model
        let user_model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files);

        // Load the model in a blocking task
        let config_name = self.config.model_name.clone();
        let (model, dimension) =
            tokio::task::spawn_blocking(move || -> Result<(TextEmbedding, usize)> {
                tracing::info!("Initializing user-defined model: {}", config_name);

                let mut model =
                    TextEmbedding::try_new_from_user_defined(user_model, Default::default())
                        .map_err(|e| EmbedError::External { source: e })?;

                // Get dimension by generating a test embedding
                let test_embeddings = model
                    .embed(vec!["test".to_string()], None)
                    .map_err(|e| EmbedError::External { source: e })?;
                let dimension = test_embeddings.first().map(|emb| emb.len()).unwrap_or(1024); // ModernBERT-large typically has 1024 dimensions

                tracing::info!(
                    "User-defined model loaded successfully. Dimension: {}",
                    dimension
                );
                Ok((model, dimension))
            })
            .await??;

        Ok((model, dimension))
    }

    /// Validate that the model is working correctly
    async fn validate_model(&self) -> Result<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| EmbedError::invalid_config("Model not initialized"))?;

        // Test the model with a simple embedding
        let test_text = "validation test";
        let model_clone = Arc::clone(model);

        let validation_result = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
            let mut model_guard = model_clone.lock().unwrap();
            model_guard
                .embed(vec![test_text.to_string()], None)
                .map_err(|e| EmbedError::External { source: e })
        })
        .await??;

        if validation_result.is_empty() {
            return Err(EmbedError::invalid_config(
                "Model validation failed: no embeddings generated",
            ));
        }

        let embedding = &validation_result[0];
        if embedding.is_empty() {
            return Err(EmbedError::invalid_config(
                "Model validation failed: empty embedding",
            ));
        }

        // Validate embedding dimension matches expected
        if embedding.len() != self.dimension {
            return Err(EmbedError::invalid_config(format!(
                "Model validation failed: expected dimension {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }

        // Check for NaN or infinite values
        for value in embedding {
            if !value.is_finite() {
                return Err(EmbedError::invalid_config(
                    "Model validation failed: non-finite values in embedding",
                ));
            }
        }

        tracing::debug!("Model validation passed for: {}", self.config.model_name);
        Ok(())
    }

    /// Clear the global model cache (useful for testing or memory management)
    pub fn clear_cache() {
        let cache = get_model_cache();
        let mut cache_guard = cache.lock().unwrap();
        cache_guard.clear();
        tracing::info!("Model cache cleared");
    }

    /// Get the number of cached models
    pub fn cache_size() -> usize {
        let cache = get_model_cache();
        let cache_guard = cache.lock().unwrap();
        cache_guard.len()
    }

    /// Convert f32 embeddings to f16
    fn convert_to_f16(&self, embeddings: Vec<Vec<f32>>) -> Vec<Vec<f16>> {
        embeddings
            .into_iter()
            .map(|embedding| {
                let mut f16_embedding: Vec<f16> =
                    embedding.into_iter().map(f16::from_f32).collect();

                // Normalize if configured
                if self.config.normalize {
                    let norm: f32 = f16_embedding
                        .iter()
                        .map(|x| x.to_f32() * x.to_f32())
                        .sum::<f32>()
                        .sqrt();
                    if norm > 0.0 {
                        for value in &mut f16_embedding {
                            *value = f16::from_f32(value.to_f32() / norm);
                        }
                    }
                }

                f16_embedding
            })
            .collect()
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    async fn embed_text(&self, text: &str) -> Result<Vec<f16>> {
        let texts = vec![text.to_string()];
        let result = self.embed_texts(&texts).await?;
        result
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::invalid_config("No embedding generated for text"))
    }

    async fn embed_texts(&self, texts: &[String]) -> Result<EmbeddingResult> {
        if texts.is_empty() {
            return Ok(EmbeddingResult::new(vec![]));
        }

        let model = self.model.as_ref().ok_or_else(|| {
            EmbedError::invalid_config("Model not initialized. Call initialize() first.")
        })?;

        tracing::debug!("Generating embeddings for {} texts", texts.len());

        // Process in batches to avoid memory issues
        let batch_size = self.config.batch_size;
        let mut all_embeddings = Vec::new();

        for chunk in texts.chunks(batch_size) {
            let chunk = chunk.to_vec();
            let model_clone = Arc::clone(model);

            let batch_embeddings = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
                tracing::debug!("Processing batch of {} texts", chunk.len());

                let mut model_guard = model_clone.lock().unwrap();
                let embeddings = model_guard
                    .embed(chunk, None)
                    .map_err(|e| EmbedError::External { source: e })?;

                Ok(embeddings)
            })
            .await??;

            // Convert f32 to f16
            let f16_embeddings = self.convert_to_f16(batch_embeddings);
            all_embeddings.extend(f16_embeddings);
        }

        tracing::debug!("Generated {} embeddings", all_embeddings.len());
        Ok(EmbeddingResult::new(all_embeddings))
    }

    fn embedding_dimension(&self) -> usize {
        self.dimension
    }

    fn provider_name(&self) -> &str {
        "fastembed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_embedding_result() {
        let embeddings = vec![
            vec![f16::from_f32(0.1), f16::from_f32(0.2), f16::from_f32(0.3)],
            vec![f16::from_f32(0.4), f16::from_f32(0.5), f16::from_f32(0.6)],
        ];
        let result = EmbeddingResult::new(embeddings);

        assert_eq!(result.len(), 2);
        assert_eq!(result.dimension, 3);
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn test_fastembed_provider_creation() {
        let temp_dir = tempdir().unwrap();
        let config = EmbedConfig::default_with_path(temp_dir.path());
        let provider = FastEmbedProvider::new(config);

        assert_eq!(provider.provider_name(), "fastembed");
        assert_eq!(provider.embedding_dimension(), 1024); // Default for ModernBERT-large
    }

    #[tokio::test]
    async fn test_modernbert_config() {
        let temp_dir = tempdir().unwrap();
        let config = EmbedConfig::modernbert_large(temp_dir.path());

        assert_eq!(config.model_name, "ModernBERT-large");
        assert_eq!(config.hf_repo(), Some("answerdotai/ModernBERT-large"));
        assert_eq!(config.hf_revision(), "main");
        assert!(config.is_huggingface_model());
    }

    #[tokio::test]
    async fn test_model_caching() -> Result<()> {
        // Clear cache before test
        FastEmbedProvider::clear_cache();
        assert_eq!(FastEmbedProvider::cache_size(), 0);

        let temp_dir = tempdir().unwrap();
        let config = EmbedConfig::default_with_path(temp_dir.path()).with_batch_size(1);

        // Create first provider - should load model
        let _provider1 = FastEmbedProvider::create(config.clone()).await?;
        assert_eq!(FastEmbedProvider::cache_size(), 1);

        // Create second provider with same config - should use cache
        let _provider2 = FastEmbedProvider::create(config).await?;
        assert_eq!(FastEmbedProvider::cache_size(), 1); // Still 1, reused from cache

        // Create provider with different config - should create new cache entry
        let config2 = EmbedConfig::default_with_path(temp_dir.path()).with_batch_size(2); // Different batch size
        let _provider3 = FastEmbedProvider::create(config2).await?;
        assert_eq!(FastEmbedProvider::cache_size(), 2);

        // Clear cache
        FastEmbedProvider::clear_cache();
        assert_eq!(FastEmbedProvider::cache_size(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_user_defined_model_configuration() {
        // Test that ModernBERT config is set up correctly for user-defined model loading
        let temp_dir = tempdir().unwrap();
        let config = EmbedConfig::modernbert_large(temp_dir.path());

        assert!(config.is_huggingface_model());
        assert_eq!(config.model_name, "ModernBERT-large");
        assert_eq!(config.hf_repo(), Some("answerdotai/ModernBERT-large"));

        // Check that paths are set up correctly
        let model_path = config.model_path();
        let onnx_path = config.onnx_model_path();
        let tokenizer_path = config.tokenizer_path();

        assert!(model_path.to_string_lossy().contains("ModernBERT-large"));
        // ONNX path should contain either model_q4.onnx (if exists) or model_quantized.onnx (fallback)
        let onnx_path_str = onnx_path.to_string_lossy();
        assert!(
            onnx_path_str.ends_with("model_q4.onnx")
                || onnx_path_str.ends_with("model_quantized.onnx")
        );
        assert!(tokenizer_path.to_string_lossy().contains("tokenizer.json"));
    }
}
