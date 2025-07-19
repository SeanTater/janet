use super::file_index::{ChunkRef, FileIndex, FileRef};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Debug, serde::Deserialize)]
pub struct BertChunkConfig {
    api_base: String,
    api_key: String,
    chunk_size_lines: usize,
    chunk_step_lines: usize,
}

/// The new trait that abstracts the analyzer.
#[async_trait]
pub trait AnalyzerTrait: Send + Sync {
    async fn analyze(&self, absolute_path: &Path) -> Result<()>;
}

/// The original Analyzer implementing the trait.
pub struct RemoteBertChunkAnalyzer {
    file_index: FileIndex,
    client: reqwest::Client,
    config: BertChunkConfig,
    gitignore: ignore::gitignore::Gitignore,
}

impl RemoteBertChunkAnalyzer {
    pub fn new(file_index: FileIndex, config: BertChunkConfig) -> Self {
        Self {
            gitignore: ignore::gitignore::Gitignore::new(&file_index.base).0,
            file_index,
            config,
            client: Client::new(),
        }
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

        Ok(Some(
            (0..lines.len())
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
                .collect(),
        ))
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
