use super::file_index::{ChunkRef, FileIndex, FileRef};
use anyhow::Result;
use async_trait::async_trait;
use janet_ai_embed::{EmbedConfig, EmbeddingProvider, FastEmbedProvider};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Debug, serde::Deserialize)]
pub struct BertChunkConfig {
    /// Base path for embedding models (e.g., "models/")
    pub model_base_path: Option<String>,
    /// Name of the embedding model to use
    pub model_name: Option<String>,
    /// Number of lines per chunk
    pub chunk_size_lines: usize,
    /// Step size between chunks (for overlapping chunks)
    pub chunk_step_lines: usize,
    /// Whether to generate embeddings for chunks
    pub generate_embeddings: bool,
}

impl Default for BertChunkConfig {
    fn default() -> Self {
        Self {
            model_base_path: Some("models".to_string()),
            model_name: Some("snowflake-arctic-embed-xs".to_string()),
            chunk_size_lines: 50,
            chunk_step_lines: 25,
            generate_embeddings: true,
        }
    }
}

/// The new trait that abstracts the analyzer.
#[async_trait]
pub trait AnalyzerTrait: Send + Sync {
    async fn analyze(&self, absolute_path: &Path) -> Result<()>;
}

/// The original Analyzer implementing the trait.
pub struct RemoteBertChunkAnalyzer {
    file_index: FileIndex,
    config: BertChunkConfig,
    gitignore: ignore::gitignore::Gitignore,
    embedding_provider: Option<FastEmbedProvider>,
}

impl RemoteBertChunkAnalyzer {
    pub fn new(file_index: FileIndex, config: BertChunkConfig) -> Self {
        Self {
            gitignore: ignore::gitignore::Gitignore::new(&file_index.base).0,
            file_index,
            config,
            embedding_provider: None,
        }
    }

    /// Initialize the embedding provider if embeddings are enabled
    pub async fn initialize_embeddings(&mut self) -> Result<()> {
        if !self.config.generate_embeddings {
            tracing::info!("Embeddings disabled in configuration");
            return Ok(());
        }

        let model_base_path = self.config.model_base_path.as_deref().unwrap_or("models");
        let model_name = self
            .config
            .model_name
            .as_deref()
            .unwrap_or("snowflake-arctic-embed-xs");

        let embed_config =
            EmbedConfig::new(model_base_path, model_name.to_string()).with_batch_size(16); // Smaller batch size for better responsiveness

        match FastEmbedProvider::create(embed_config).await {
            Ok(provider) => {
                tracing::info!("Embedding provider initialized successfully");
                self.embedding_provider = Some(provider);
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to initialize embedding provider: {}. Continuing without embeddings.",
                    e
                );
                // Don't fail the entire analyzer if embeddings can't be initialized
            }
        }

        Ok(())
    }

    /// Create and initialize a new analyzer with embeddings
    pub async fn create_with_embeddings(
        file_index: FileIndex,
        config: BertChunkConfig,
    ) -> Result<Self> {
        let mut analyzer = Self::new(file_index, config);
        analyzer.initialize_embeddings().await?;
        Ok(analyzer)
    }

    pub async fn chunk_file(&self, relative_path: &Path) -> Result<Option<Vec<ChunkRef>>> {
        // Naive implementation of chunking
        let loc = &self.file_index.base.join(relative_path);
        let metadata = tokio::fs::metadata(loc).await?;
        if self
            .gitignore
            .matched_path_or_any_parents(relative_path, metadata.is_dir())
            .is_ignore()
        {
            return Ok(None);
        }
        if metadata.len() > 1 << 20 {
            return Ok(None);
        }
        let buf = tokio::fs::read(loc).await?;
        let file_hash = *blake3::hash(&buf).as_bytes();
        let buf = String::from_utf8_lossy(&buf);
        let lines = buf.lines().collect::<Vec<_>>();

        // Store file reference in database
        let file_ref = FileRef {
            relative_path: relative_path.to_string_lossy().to_string(),
            content: buf.as_bytes().to_vec(),
            hash: file_hash,
        };
        self.file_index.upsert_file(&file_ref).await?;

        // Create chunks
        let mut chunks: Vec<ChunkRef> = (0..lines.len())
            .step_by(self.config.chunk_step_lines)
            .map(|line_start| {
                let line_end = (line_start + self.config.chunk_size_lines).min(lines.len());
                let content = lines[line_start..line_end].join("\n");
                ChunkRef {
                    id: None,
                    file_hash,
                    relative_path: relative_path.to_string_lossy().to_string(),
                    line_start,
                    line_end,
                    content,
                    embedding: None,
                }
            })
            .collect();

        // Generate embeddings if provider is available
        if let Some(provider) = &self.embedding_provider {
            tracing::debug!("Generating embeddings for {} chunks", chunks.len());

            let texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();

            match provider.embed_texts(&texts).await {
                Ok(embedding_result) => {
                    let num_embeddings = embedding_result.len();
                    for (chunk, embedding) in chunks.iter_mut().zip(embedding_result.embeddings) {
                        chunk.embedding = Some(embedding);
                    }
                    tracing::debug!("Successfully generated {} embeddings", num_embeddings);
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to generate embeddings for {}: {}",
                        relative_path.display(),
                        e
                    );
                    // Continue without embeddings rather than failing
                }
            }
        }

        Ok(Some(chunks))
    }
}

#[async_trait]
impl AnalyzerTrait for RemoteBertChunkAnalyzer {
    async fn analyze(&self, relative_path: &Path) -> Result<()> {
        tracing::info!("Indexing {}", relative_path.display());

        // Skip if not a file or if in gitignore
        let absolute_path = self.file_index.base.join(relative_path);
        if !absolute_path.is_file() {
            return Ok(());
        }

        // Chunk the file
        if let Some(chunks) = self.chunk_file(relative_path).await? {
            // Store chunks in database
            self.file_index.upsert_chunks(&chunks).await?;
            tracing::debug!(
                "Stored {} chunks for {}",
                chunks.len(),
                relative_path.display()
            );
        } else {
            tracing::debug!(
                "Skipped file {} (ignored or too large)",
                relative_path.display()
            );
        }

        Ok(())
    }
}

/// A new mock analyzer for testing.
/// It records every path passed to `analyze` in an internal vector.
pub struct MockAnalyzer {
    pub calls: Arc<Mutex<Vec<PathBuf>>>,
}

impl Default for MockAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl MockAnalyzer {
    pub fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl AnalyzerTrait for MockAnalyzer {
    async fn analyze(&self, relative_path: &Path) -> Result<()> {
        let mut calls = self.calls.lock().unwrap();
        calls.push(relative_path.to_path_buf());
        Ok(())
    }
}
