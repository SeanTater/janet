use super::file_index::FileIndex;
use anyhow::Result;
use async_trait::async_trait;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// The new trait that abstracts the analyzer.
#[async_trait]
pub trait AnalyzerTrait: Send + Sync {
    async fn analyze(&self, absolute_path: &Path) -> Result<()>;
}

/// The original Analyzer implementing the trait.
struct Analyzer {
    file_index: FileIndex,
}

#[async_trait]
impl AnalyzerTrait for Analyzer {
    async fn analyze(&self, absolute_path: &Path) -> Result<()> {
        tracing::info!("Would have indexed {}", absolute_path.display());
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
    async fn analyze(&self, absolute_path: &Path) -> Result<()> {
        let mut calls = self.calls.lock().unwrap();
        calls.push(absolute_path.to_path_buf());
        Ok(())
    }
}
