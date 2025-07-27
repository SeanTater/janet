//! Storage abstraction layer for janet-ai-retriever
//!
//! This module provides trait-based abstractions for storing and retrieving code chunks
//! and their embeddings. It separates the storage concerns from the retrieval logic,
//! allowing for different storage backends while maintaining a consistent API.
//!
//! ## Key Components
//!
//! - **ChunkStore**: Text storage and retrieval operations
//! - **EmbeddingStore**: Vector similarity search operations
//! - **CombinedStore**: Unified interface combining both stores
//! - **Data Types**: Chunk, File, and metadata structures
//!
//! ## Architecture
//!
//! ```text
//! ChunkStore ─┐
//!             ├─ CombinedStore ── SQLiteStore (concrete implementation)
//! EmbeddingStore ─┘
//! ```
//!
//! ## Usage
//!

use anyhow::Result;
use async_trait::async_trait;

pub mod sqlite_store;

/// Blake3 hash identifying a unique file (32 bytes).
pub type FileHash = [u8; 32];

/// Database ID for a text chunk.
pub type ChunkId = i64;

/// Code chunk with content and metadata. See module docs for usage examples.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: Option<ChunkId>,
    pub file_hash: FileHash,
    pub relative_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
    pub embedding: Option<Vec<half::f16>>,
}

/// File metadata for the index. See module docs for details.
#[derive(Debug, Clone)]
pub struct File {
    pub hash: FileHash,
    pub relative_path: String,
    pub size: usize,
}

/// Query filter for chunk searches. See module docs for usage examples.
#[derive(Debug, Clone, Default)]
pub struct ChunkFilter {
    pub file_hash: Option<FileHash>,
    pub path_prefix: Option<String>,
    pub has_embedding: Option<bool>,
}

/// Chunk metadata without content. See module docs for details.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChunkMetadata {
    pub id: ChunkId,
    pub file_hash: FileHash,
    pub relative_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub has_embedding: bool,
}

/// Text storage operations for code chunks. See module docs for usage examples.
#[async_trait]
pub trait ChunkStore: Send + Sync {
    /// Insert new chunks and return their IDs
    async fn insert_chunks(&self, chunks: Vec<Chunk>) -> Result<Vec<ChunkId>>;

    /// Get a specific chunk by ID
    async fn get_chunk(&self, id: ChunkId) -> Result<Option<Chunk>>;

    /// Update an existing chunk
    async fn update_chunk(&self, id: ChunkId, chunk: Chunk) -> Result<()>;

    /// Delete chunks for a specific file
    async fn delete_chunks(&self, file_hash: FileHash) -> Result<usize>;

    /// List chunks matching filter criteria
    async fn list_chunks(&self, filter: ChunkFilter) -> Result<Vec<ChunkMetadata>>;

    /// Get all chunks for a file
    async fn get_file_chunks(&self, file_hash: FileHash) -> Result<Vec<Chunk>>;

    /// Search for chunks containing the specified text
    async fn search_text(&self, search_term: &str, case_sensitive: bool) -> Result<Vec<Chunk>>;
}

/// Vector similarity search operations. See module docs for details.
#[async_trait]
pub trait EmbeddingStore: Send + Sync {
    /// Store embeddings for chunks
    async fn store_embeddings(
        &self,
        chunk_ids: Vec<ChunkId>,
        embeddings: Vec<Vec<half::f16>>,
    ) -> Result<()>;

    /// Search for similar chunks using vector similarity
    async fn search_similar(
        &self,
        query: Vec<half::f16>,
        limit: usize,
        threshold: Option<half::f16>,
    ) -> Result<Vec<(ChunkId, half::f16)>>;

    /// Delete embeddings for specific chunks
    async fn delete_embeddings(&self, chunk_ids: Vec<ChunkId>) -> Result<()>;

    /// Get embedding for a specific chunk
    async fn get_embedding(&self, chunk_id: ChunkId) -> Result<Option<Vec<half::f16>>>;
}

/// Unified store combining text and vector operations. See module docs for details.
#[async_trait]
pub trait CombinedStore: ChunkStore + EmbeddingStore + Send + Sync {
    /// Search for similar chunks and return full chunk data
    async fn search_chunks(
        &self,
        query: Vec<half::f16>,
        limit: usize,
        threshold: Option<half::f16>,
    ) -> Result<Vec<(Chunk, half::f16)>>;
}
