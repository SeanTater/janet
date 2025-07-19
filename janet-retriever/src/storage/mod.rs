use anyhow::Result;
use async_trait::async_trait;

pub mod sqlite_store;

pub type FileHash = [u8; 32];
pub type ChunkId = i64;

/// Represents a code chunk with metadata
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: Option<ChunkId>,
    pub file_hash: FileHash,
    pub relative_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
}

/// Represents a file in the index
#[derive(Debug, Clone)]
pub struct File {
    pub hash: FileHash,
    pub relative_path: String,
    pub size: usize,
}

/// Filter criteria for chunk queries
#[derive(Debug, Clone, Default)]
pub struct ChunkFilter {
    pub file_hash: Option<FileHash>,
    pub path_prefix: Option<String>,
    pub has_embedding: Option<bool>,
}

/// Metadata about a chunk (without the full content)
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub id: ChunkId,
    pub file_hash: FileHash,
    pub relative_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub has_embedding: bool,
}

/// Trait for storing and retrieving code chunks
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
}

/// Trait for vector similarity search
#[async_trait]
pub trait EmbeddingStore: Send + Sync {
    /// Store embeddings for chunks
    async fn store_embeddings(&self, chunk_ids: Vec<ChunkId>, embeddings: Vec<Vec<f32>>) -> Result<()>;
    
    /// Search for similar chunks using vector similarity
    async fn search_similar(&self, query: Vec<f32>, limit: usize, threshold: Option<f32>) -> Result<Vec<(ChunkId, f32)>>;
    
    /// Delete embeddings for specific chunks
    async fn delete_embeddings(&self, chunk_ids: Vec<ChunkId>) -> Result<()>;
    
    /// Get embedding for a specific chunk
    async fn get_embedding(&self, chunk_id: ChunkId) -> Result<Option<Vec<f32>>>;
}

/// Combined store that implements both chunk storage and embedding search
#[async_trait]
pub trait CombinedStore: ChunkStore + EmbeddingStore + Send + Sync {
    /// Search for similar chunks and return full chunk data
    async fn search_chunks(&self, query: Vec<f32>, limit: usize, threshold: Option<f32>) -> Result<Vec<(Chunk, f32)>>;
}