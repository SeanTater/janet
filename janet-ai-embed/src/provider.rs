//! Embedding provider implementations

use crate::config::EmbedConfig;
use crate::downloader::download_model;
use crate::error::{EmbedError, Result};
use async_trait::async_trait;
use fastembed::{
    EmbeddingModel, InitOptions, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use fnv::FnvHasher;
use half::f16;
use serde_json;
use std::collections::HashMap;
use std::hash::Hasher;
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
    /// Create a new embedding result from a vector of f16 embeddings.
    ///
    /// The dimension is automatically inferred from the first embedding vector.
    /// If the embeddings vector is empty, dimension defaults to 0.
    ///
    /// # Arguments
    /// * `embeddings` - Vector of embedding vectors, where each inner vector represents
    ///   the embedding for one input text
    ///
    /// # Example
    pub fn new(embeddings: Vec<Vec<f16>>) -> Self {
        let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);
        Self {
            embeddings,
            dimension,
        }
    }

    /// Returns the number of embedding vectors in this result.
    ///
    /// # Returns
    /// The count of embedding vectors (i.e., the number of input texts that were embedded)
    ///
    /// # Example
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Returns `true` if this result contains no embedding vectors.
    ///
    /// # Returns
    /// `true` if there are no embeddings, `false` otherwise
    ///
    /// # Example
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
#[derive(Clone)]
pub struct FastEmbedProvider {
    config: EmbedConfig,
    model: Option<Arc<Mutex<TextEmbedding>>>,
    dimension: usize,
}

impl std::fmt::Debug for FastEmbedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedProvider")
            .field("config", &self.config)
            .field("model", &self.model.is_some())
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl FastEmbedProvider {
    /// Creates a new uninitialized provider. See module docs for usage patterns and initialization.
    pub fn new(config: EmbedConfig) -> Self {
        Self {
            config,
            model: None,
            dimension: 1024, // Default dimension for ModernBERT-large
        }
    }

    /// Downloads and loads the embedding model with caching. See module docs for details.
    pub async fn initialize(&mut self) -> Result<()> {
        tracing::info!(
            "Initializing FastEmbed provider for model: {}",
            self.config.model_name().to_string()
        );

        // Create a cache key based on the model configuration
        let cache_key = self.create_cache_key();

        // Check for old cache format and clear if needed
        {
            let cache = get_model_cache().lock().unwrap();
            let has_old_entries = cache.keys().any(|key| !key.starts_with("v1:"));
            if has_old_entries {
                drop(cache); // Release lock before clearing
                Self::clear_cache();
                tracing::info!("Cleared cache due to version upgrade");
            }
        }

        // Check if model is already cached
        let cached_data = {
            let cache = get_model_cache().lock().unwrap();
            cache
                .get(&cache_key)
                .map(|(model, dim)| (Arc::clone(model), *dim))
        };

        if let Some((cached_model, cached_dimension)) = cached_data {
            tracing::info!("Using cached model for: {}", self.config.model_name());
            self.model = Some(cached_model);
            self.dimension = cached_dimension;
            return self.validate_model().await;
        }

        // Check if this is a HuggingFace model that needs downloading
        if self.config.is_huggingface_model() {
            tracing::info!(
                "Downloading HuggingFace model: {}",
                self.config.model_name()
            );
            download_model(&self.config).await?;

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
            tracing::info!(
                "Using built-in fastembed model: {}",
                self.config.model_name()
            );

            // Load model in a blocking task
            let config = self.config.clone();
            let (model, dimension) =
                tokio::task::spawn_blocking(move || -> Result<(TextEmbedding, usize)> {
                    tracing::info!("Loading embedding model: {}", config.model_name());

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

    /// Creates and initializes a provider in one step. See module docs for usage patterns.
    pub async fn create(config: EmbedConfig) -> Result<Self> {
        let mut provider = Self::new(config);
        provider.initialize().await?;
        Ok(provider)
    }

    /// Create a cache key based on the model configuration
    fn create_cache_key(&self) -> String {
        // Serialize entire config to deterministic JSON
        let config_json =
            serde_json::to_string(&self.config).expect("Config should always serialize");

        // Hash with FNV for deterministic, fast hashing
        let mut hasher = FnvHasher::default();
        hasher.write(b"v1:"); // Version prefix
        hasher.write(config_json.as_bytes());

        format!("v1:{:x}", hasher.finish())
    }

    /// Load a user-defined ONNX model from downloaded HuggingFace files
    async fn load_user_defined_model(&self) -> Result<(TextEmbedding, usize)> {
        tracing::info!("Loading user-defined model: {}", self.config.model_name());

        // Read all required files
        let onnx_file = fs::read(self.config.onnx_model_path())
            .await
            .map_err(|e| EmbedError::Io { source: e })?;

        let tokenizer_config = self.config.tokenizer_config();

        let tokenizer_file = fs::read(&tokenizer_config.tokenizer_path)
            .await
            .map_err(|e| EmbedError::Io { source: e })?;

        let config_file = fs::read(&tokenizer_config.config_path)
            .await
            .map_err(|e| EmbedError::Io { source: e })?;

        let special_tokens_map_file = fs::read(&tokenizer_config.special_tokens_map_path)
            .await
            .map_err(|e| EmbedError::Io { source: e })?;

        // Check if tokenizer_config.json exists, create a minimal one if not
        let tokenizer_config_file = if let Some(path) = &tokenizer_config.tokenizer_config_path {
            if path.exists() {
                fs::read(path)
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
            }
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
        let config_name = self.config.model_name().to_string();
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

        tracing::debug!("Model validation passed for: {}", self.config.model_name());
        Ok(())
    }

    /// Clears the global model cache. See module docs for caching details.
    pub fn clear_cache() {
        let cache = get_model_cache();
        let mut cache_guard = cache.lock().unwrap();
        cache_guard.clear();
        tracing::info!("Model cache cleared");
    }

    /// Returns the number of cached models. See module docs for caching details.
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

                // Always normalize
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
        let batch_size = 16;
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

    #[test]
    fn test_embedding_result() {
        let embeddings = vec![
            vec![f16::from_f32(0.1), f16::from_f32(0.2), f16::from_f32(0.3)],
            vec![f16::from_f32(0.4), f16::from_f32(0.5), f16::from_f32(0.6)],
        ];
        let result = EmbeddingResult::new(embeddings);

        assert_eq!(result.len(), 2);
        assert_eq!(result.dimension, 3);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_fastembed_provider_creation() {
        let config = EmbedConfig::default();
        let provider = FastEmbedProvider::new(config);

        assert_eq!(provider.provider_name(), "fastembed");
        assert_eq!(provider.embedding_dimension(), 1024); // Default for ModernBERT-large
    }

    #[tokio::test]
    async fn test_modernbert_config() {
        let config = EmbedConfig::modernbert_large();

        assert_eq!(config.model_name(), "ModernBERT-large");
        assert_eq!(config.hf_repo(), Some("answerdotai/ModernBERT-large"));
        assert_eq!(config.hf_revision(), "main");
        assert!(config.is_huggingface_model());
    }

    #[tokio::test]
    #[ignore] // Integration test: Downloads real ModernBERT model, tests embeddings - run with: cargo test test_modernbert_download_and_embedding -- --ignored
    async fn test_modernbert_download_and_embedding() -> Result<()> {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init()
            .ok(); // Ignore if already initialized

        println!("🧪 Testing ModernBERT-large download and embedding generation...");
        println!("📂 Will use real cache directory: $HOME/.janet/models");

        // Clear any existing cache to ensure clean test
        FastEmbedProvider::clear_cache();
        assert_eq!(FastEmbedProvider::cache_size(), 0);

        // Create ModernBERT config
        let config = EmbedConfig::modernbert_large();
        println!("📋 Config: {config:?}");

        // Verify config properties
        assert_eq!(config.model_name(), "ModernBERT-large");
        assert!(config.is_huggingface_model());
        assert_eq!(config.hf_repo(), Some("answerdotai/ModernBERT-large"));
        assert_eq!(config.hf_revision(), "main");

        let expected_model_path = config.model_path();
        println!("📁 Expected model path: {}", expected_model_path.display());

        // Check if model already exists (may have been downloaded by previous test runs)
        let model_existed_initially = expected_model_path.exists();
        if model_existed_initially {
            println!("ℹ️  Model already exists (cached from previous run)");
        } else {
            println!("📦 Model not found, will download");
        }

        // Create provider - this should trigger download
        println!("⬇️  Creating provider (will download model)...");
        let start_time = std::time::Instant::now();

        let provider = FastEmbedProvider::create(config.clone()).await?;

        let download_time = start_time.elapsed();
        println!("✅ Provider created in {:.2}s", download_time.as_secs_f64());

        // Verify model was cached
        assert_eq!(FastEmbedProvider::cache_size(), 1);

        // Verify model files exist (either downloaded now or from previous runs)
        assert!(
            expected_model_path.exists(),
            "Model directory should exist after provider creation"
        );
        assert!(
            config.onnx_model_path().exists(),
            "ONNX model file should exist"
        );

        let tokenizer_config = config.tokenizer_config();
        assert!(
            tokenizer_config.tokenizer_path.exists(),
            "Tokenizer file should exist"
        );
        assert!(
            tokenizer_config.config_path.exists(),
            "Config file should exist"
        );
        assert!(
            tokenizer_config.special_tokens_map_path.exists(),
            "Special tokens map should exist"
        );

        // Test provider properties
        assert_eq!(provider.provider_name(), "fastembed");
        assert_eq!(provider.embedding_dimension(), 1024); // ModernBERT-large dimension

        // Test single text embedding
        println!("🔤 Testing single text embedding...");
        let test_text = "ModernBERT is a state-of-the-art transformer model for efficient text embedding generation.";

        let embedding = provider.embed_text(test_text).await?;
        assert_eq!(
            embedding.len(),
            1024,
            "Single embedding should have correct dimension"
        );

        // Verify embedding values are reasonable (not all zeros, finite)
        assert!(
            embedding.iter().any(|&x| x.to_f32() != 0.0),
            "Embedding should not be all zeros"
        );
        assert!(
            embedding.iter().all(|&x| x.to_f32().is_finite()),
            "All embedding values should be finite"
        );

        // Test batch embedding
        println!("📚 Testing batch embedding...");
        let test_texts = vec![
            "Machine learning models process natural language efficiently.".to_string(),
            "Deep neural networks enable semantic understanding of text.".to_string(),
            "Transformer architectures revolutionized natural language processing.".to_string(),
        ];

        let batch_result = provider.embed_texts(&test_texts).await?;
        assert_eq!(
            batch_result.len(),
            3,
            "Should generate embeddings for all input texts"
        );
        assert_eq!(
            batch_result.dimension, 1024,
            "Batch result should have correct dimension"
        );

        // Verify all batch embeddings
        for (i, embedding) in batch_result.embeddings.iter().enumerate() {
            assert_eq!(
                embedding.len(),
                1024,
                "Batch embedding {i} should have correct dimension"
            );
            assert!(
                embedding.iter().any(|&x| x.to_f32() != 0.0),
                "Batch embedding {i} should not be all zeros"
            );
            assert!(
                embedding.iter().all(|&x| x.to_f32().is_finite()),
                "All values in batch embedding {i} should be finite"
            );
        }

        // Test semantic similarity - related texts should have higher similarity
        let emb1 = &batch_result.embeddings[0]; // ML models
        let emb2 = &batch_result.embeddings[1]; // Deep neural networks
        let emb3 = &batch_result.embeddings[2]; // Transformers

        // Calculate cosine similarities (embeddings are already normalized)
        let sim_1_2: f32 = emb1
            .iter()
            .zip(emb2.iter())
            .map(|(a, b)| a.to_f32() * b.to_f32())
            .sum();
        let sim_1_3: f32 = emb1
            .iter()
            .zip(emb3.iter())
            .map(|(a, b)| a.to_f32() * b.to_f32())
            .sum();
        let sim_2_3: f32 = emb2
            .iter()
            .zip(emb3.iter())
            .map(|(a, b)| a.to_f32() * b.to_f32())
            .sum();

        println!("🔍 Semantic similarities:");
        println!("   ML ↔ Neural Networks: {sim_1_2:.3}");
        println!("   ML ↔ Transformers: {sim_1_3:.3}");
        println!("   Neural Networks ↔ Transformers: {sim_2_3:.3}");

        // All similarities should be reasonably high (> 0.3) since texts are related
        assert!(
            sim_1_2 > 0.3,
            "Related texts should have reasonable similarity: {sim_1_2}"
        );
        assert!(
            sim_1_3 > 0.3,
            "Related texts should have reasonable similarity: {sim_1_3}"
        );
        assert!(
            sim_2_3 > 0.3,
            "Related texts should have reasonable similarity: {sim_2_3}"
        );

        // Test caching - second provider with same config should reuse cached model
        println!("💾 Testing model caching...");
        let start_time = std::time::Instant::now();

        let provider2 = FastEmbedProvider::create(config).await?;

        let cache_time = start_time.elapsed();
        println!(
            "✅ Second provider created in {:.3}s (should be much faster)",
            cache_time.as_secs_f64()
        );

        // Should still be only 1 cached model
        assert_eq!(FastEmbedProvider::cache_size(), 1);

        // Both providers should work identically
        let embedding2 = provider2.embed_text(test_text).await?;
        assert_eq!(embedding.len(), embedding2.len());

        // Clean up cache
        FastEmbedProvider::clear_cache();
        assert_eq!(FastEmbedProvider::cache_size(), 0);

        println!("🎉 ModernBERT integration test completed successfully!");
        println!("📊 Model location: {}", expected_model_path.display());
        println!(
            "🔧 To run this test: cargo test test_modernbert_download_and_embedding -- --ignored"
        );
        println!(
            "ℹ️  Note: This test uses the real cache directory for authentic integration testing"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_user_defined_model_configuration() {
        // Test that ModernBERT config is set up correctly for user-defined model loading
        let config = EmbedConfig::modernbert_large();

        assert!(config.is_huggingface_model());
        assert_eq!(config.model_name(), "ModernBERT-large");
        assert_eq!(config.hf_repo(), Some("answerdotai/ModernBERT-large"));

        // Check that paths are set up correctly
        let model_path = config.model_path();
        let onnx_path = config.onnx_model_path();
        let tokenizer_path = &config.tokenizer_config().tokenizer_path;

        assert!(model_path.to_string_lossy().contains("ModernBERT-large"));
        // ONNX path should contain either model_q4.onnx (if exists) or model_quantized.onnx (fallback)
        let onnx_path_str = onnx_path.to_string_lossy();
        assert!(
            onnx_path_str.ends_with("model_q4.onnx")
                || onnx_path_str.ends_with("model_quantized.onnx")
        );
        assert!(tokenizer_path.to_string_lossy().contains("tokenizer.json"));
    }

    #[tokio::test]
    async fn test_cache_key_generation() {
        // Test that same config produces same cache key
        let config1 = EmbedConfig::default();
        let provider1 = FastEmbedProvider::new(config1.clone());
        let key1 = provider1.create_cache_key();

        let provider2 = FastEmbedProvider::new(config1);
        let key2 = provider2.create_cache_key();

        assert_eq!(key1, key2, "Same config should produce same cache key");
        assert!(
            key1.starts_with("v1:"),
            "Cache key should have version prefix"
        );

        // Test that different configs produce different cache keys
        let config_different_batch = EmbedConfig::default();
        let provider3 = FastEmbedProvider::new(config_different_batch);
        let key3 = provider3.create_cache_key();

        // Since batch_size is now static, these configs are identical
        assert_eq!(key1, key3, "Same configs should produce same cache key");

        // Test with different model name
        let config_different_model = EmbedConfig::new("different-model");
        let provider4 = FastEmbedProvider::new(config_different_model);
        let key4 = provider4.create_cache_key();

        assert_ne!(
            key1, key4,
            "Different model name should produce different cache key"
        );

        // Since model_base_path is now inferred, test with different model configurations
        // Test local vs HuggingFace variant with different model name
        let config_local = EmbedConfig::new("some-local-model");
        let provider5 = FastEmbedProvider::new(config_local);
        let key5 = provider5.create_cache_key();

        assert_ne!(
            key1, key5,
            "Different model configurations should produce different cache key"
        );

        // Test deterministic behavior - same config should always produce same key
        let config_deterministic = EmbedConfig::default();
        let keys: Vec<String> = (0..5)
            .map(|_| {
                let provider = FastEmbedProvider::new(config_deterministic.clone());
                provider.create_cache_key()
            })
            .collect();

        assert!(
            keys.windows(2).all(|w| w[0] == w[1]),
            "Cache key generation should be deterministic"
        );
    }
}
