//! Text chunking strategies for breaking files into searchable segments.
//!
//! This module provides intelligent text chunking using janet-ai-context to break
//! source code files into appropriately-sized segments for embedding and search.
//! It handles language-specific delimiters and maintains semantic boundaries.
//!
//! ## Key Components
//!
//! - **ChunkingStrategy**: Main interface for chunking files
//! - **ChunkingConfig**: Configuration for chunk size and behavior
//! - **Integration**: Uses janet-ai-context for intelligent text segmentation
//!
//! ## Features
//!
//! ### Language-Aware Chunking
//! - Respects function and class boundaries
//! - Uses appropriate delimiters for different programming languages
//! - Preserves logical code structure in chunks
//!
//! ### Configurable Chunk Sizes
//! - Character-based chunk size limits
//! - Automatic size adjustment for long logical units
//! - Balance between search granularity and context preservation
//!
//! ### Context Preservation
//! - Maintains code context within chunks
//! - Avoids splitting mid-statement or mid-function
//! - Repository name context for better search relevance
//!
//! ## Usage
//!
//! ```rust,no_run
//! use janet_ai_retriever::retrieval::chunking_strategy::{ChunkingStrategy, ChunkingConfig};
//! use std::path::Path;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = ChunkingConfig {
//!     max_chunk_size: 1500,
//!     repo_name: "my-project".to_string(),
//! };
//!
//! let strategy = ChunkingStrategy::new(config);
//!
//! // Chunk a Rust file
//! let content = std::fs::read_to_string("src/main.rs")?;
//! let chunks = strategy.chunk_content(Path::new("src/main.rs"), &content)?;
//!
//! for chunk in chunks {
//!     println!("Chunk sequence: {}", chunk.sequence);
//!     println!("Content: {}", chunk.chunk_text);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Chunk Size Considerations
//!
//! ### Small Chunks (500-1000 chars)
//! - **Pros**: More precise search results, faster embedding generation
//! - **Cons**: May break logical code units, less context for search
//! - **Best for**: Large codebases, fine-grained search
//!
//! ### Medium Chunks (1000-2000 chars)
//! - **Pros**: Good balance of precision and context
//! - **Cons**: Default choice for most use cases
//! - **Best for**: General purpose code search
//!
//! ### Large Chunks (2000+ chars)
//! - **Pros**: Maximum context preservation, fewer database entries
//! - **Cons**: Less precise search, slower processing
//! - **Best for**: Documentation, architectural search
//!
//! ## Integration with janet-ai-context
//!
//! This module leverages janet-ai-context's `TextContextBuilder` for intelligent
//! segmentation that respects programming language syntax and semantics.

use anyhow::Result;
use janet_ai_context::{TextChunk, create_builder_for_path};
use std::path::Path;

/// Configuration for text chunking behavior. See module docs for usage examples.
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Maximum size of each chunk in characters
    pub max_chunk_size: usize,
    /// Repository name for context
    pub repo_name: String,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 2000,
            repo_name: "unknown".to_string(),
        }
    }
}

impl ChunkingConfig {
    /// Creates config with default chunk size and specified repository name.
    pub fn new(repo_name: String) -> Self {
        Self {
            max_chunk_size: 2000,
            repo_name,
        }
    }

    /// Builder method to set maximum chunk size in characters.
    pub fn with_max_chunk_size(mut self, max_chunk_size: usize) -> Self {
        self.max_chunk_size = max_chunk_size;
        self
    }
}

/// Text chunking strategy using janet-ai-context. See module docs for examples.
#[derive(Clone, Debug)]
pub struct ChunkingStrategy {
    config: ChunkingConfig,
}

impl ChunkingStrategy {
    /// Create a new chunking strategy with the given configuration
    /// Creates a new chunking strategy with the specified configuration.
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Chunk a file's content using janet-ai-context
    /// Chunks file content using language-aware delimiters. See module docs for details.
    pub fn chunk_content(&self, file_path: &Path, content: &str) -> Result<Vec<TextChunk>> {
        // Use janet-ai-context to create the appropriate builder for this file type
        let builder = create_builder_for_path(
            self.config.repo_name.clone(),
            file_path,
            Some(self.config.max_chunk_size),
        );

        let chunks = builder.get_chunks(content);

        tracing::debug!(
            "Chunked {} into {} chunks (max size: {})",
            file_path.display(),
            chunks.len(),
            self.config.max_chunk_size
        );

        Ok(chunks)
    }

    /// Check if a file should be indexed based on its path
    /// Returns whether a file path should be indexed based on its extension.
    pub fn should_index_file(&self, file_path: &Path) -> bool {
        // Skip hidden files and common binary/generated file extensions
        if let Some(filename) = file_path.file_name().and_then(|n| n.to_str()) {
            if filename.starts_with('.') {
                return false;
            }
        }

        match file_path.extension().and_then(|ext| ext.to_str()) {
            // Text and code files we want to index
            Some("rs") | Some("py") | Some("js") | Some("ts") | Some("jsx") | Some("tsx")
            | Some("go") | Some("java") | Some("c") | Some("cpp") | Some("h") | Some("hpp")
            | Some("md") | Some("markdown") | Some("txt") | Some("toml") | Some("yaml")
            | Some("yml") | Some("json") => true,

            // Binary and generated files we skip
            Some("exe") | Some("dll") | Some("so") | Some("dylib") | Some("bin") | Some("png")
            | Some("jpg") | Some("jpeg") | Some("gif") | Some("ico") | Some("wasm")
            | Some("lock") => false,

            // Files without extensions - check if they're common text files
            None => {
                if let Some(filename) = file_path.file_name().and_then(|n| n.to_str()) {
                    matches!(
                        filename,
                        "README" | "CHANGELOG" | "LICENSE" | "Makefile" | "Dockerfile"
                    )
                } else {
                    false
                }
            }

            // Unknown extensions - default to indexing
            Some(_) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_chunking_strategy() {
        let config = ChunkingConfig::new("test_repo".to_string());
        let strategy = ChunkingStrategy::new(config);

        let rust_content = r#"
use std::collections::HashMap;

pub struct MyStruct {
    field: String,
}

impl MyStruct {
    pub fn new(field: String) -> Self {
        Self { field }
    }

    pub fn get_field(&self) -> &str {
        &self.field
    }
}

fn main() {
    let s = MyStruct::new("test".to_string());
    println!("{}", s.get_field());
}
"#;

        let rust_path = PathBuf::from("src/main.rs");
        let chunks = strategy.chunk_content(&rust_path, rust_content).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].repo, "test_repo");
        assert_eq!(chunks[0].path, "src/main.rs");

        // Verify that all chunks combined reconstruct the original content
        let reconstructed: String = chunks.iter().map(|c| c.chunk_text.as_str()).collect();
        assert_eq!(reconstructed, rust_content);
    }

    #[test]
    fn test_should_index_file() {
        let config = ChunkingConfig::new("test".to_string());
        let strategy = ChunkingStrategy::new(config);

        assert!(strategy.should_index_file(Path::new("src/lib.rs")));
        assert!(strategy.should_index_file(Path::new("README.md")));
        assert!(strategy.should_index_file(Path::new("config.json")));
        assert!(strategy.should_index_file(Path::new("README")));

        assert!(!strategy.should_index_file(Path::new("binary.exe")));
        assert!(!strategy.should_index_file(Path::new("image.png")));
        assert!(!strategy.should_index_file(Path::new(".hidden")));
        assert!(!strategy.should_index_file(Path::new("Cargo.lock")));
    }
}
