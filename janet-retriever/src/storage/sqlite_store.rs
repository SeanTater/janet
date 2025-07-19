use super::{Chunk, ChunkStore, EmbeddingStore, CombinedStore, ChunkFilter, ChunkMetadata, FileHash, ChunkId};
use crate::retrieval::file_index::{FileIndex, ChunkRef, FileRef};
use anyhow::Result;
use async_trait::async_trait;

/// Implementation of storage traits for SQLite-based FileIndex
pub struct SqliteStore {
    file_index: FileIndex,
}

impl SqliteStore {
    pub fn new(file_index: FileIndex) -> Self {
        Self { file_index }
    }
}

// Helper functions to convert between internal types and public API types
impl SqliteStore {
    fn chunk_ref_to_chunk(chunk_ref: ChunkRef) -> Chunk {
        Chunk {
            id: chunk_ref.id,
            file_hash: chunk_ref.file_hash,
            relative_path: chunk_ref.relative_path,
            line_start: chunk_ref.line_start,
            line_end: chunk_ref.line_end,
            content: chunk_ref.content,
            embedding: chunk_ref.embedding,
        }
    }

    fn chunk_to_chunk_ref(chunk: Chunk) -> ChunkRef {
        ChunkRef {
            id: chunk.id,
            file_hash: chunk.file_hash,
            relative_path: chunk.relative_path,
            line_start: chunk.line_start,
            line_end: chunk.line_end,
            content: chunk.content,
            embedding: chunk.embedding,
        }
    }
}

#[async_trait]
impl ChunkStore for SqliteStore {
    async fn insert_chunks(&self, chunks: Vec<Chunk>) -> Result<Vec<ChunkId>> {
        let chunk_refs: Vec<ChunkRef> = chunks.into_iter().map(Self::chunk_to_chunk_ref).collect();
        self.file_index.upsert_chunks(&chunk_refs).await?;
        
        // Since upsert doesn't return IDs, we need to query them back
        // This is a limitation of the current FileIndex API - in a real implementation
        // you'd want to return the IDs from the upsert operation
        let mut ids = Vec::new();
        for chunk_ref in chunk_refs {
            // Get chunks for this file and find the matching one
            let file_chunks = self.file_index.get_chunks(&chunk_ref.file_hash).await?;
            for file_chunk in file_chunks {
                if file_chunk.line_start == chunk_ref.line_start && 
                   file_chunk.line_end == chunk_ref.line_end {
                    if let Some(id) = file_chunk.id {
                        ids.push(id);
                        break;
                    }
                }
            }
        }
        Ok(ids)
    }

    async fn get_chunk(&self, id: ChunkId) -> Result<Option<Chunk>> {
        // The current FileIndex doesn't support getting by ID directly
        // This would need to be added to the FileIndex implementation
        unimplemented!("get_chunk by ID not yet implemented in FileIndex")
    }

    async fn update_chunk(&self, _id: ChunkId, _chunk: Chunk) -> Result<()> {
        // Would need to be implemented in FileIndex
        unimplemented!("update_chunk not yet implemented in FileIndex")
    }

    async fn delete_chunks(&self, file_hash: FileHash) -> Result<usize> {
        self.file_index.delete_chunks(&file_hash).await
    }

    async fn list_chunks(&self, filter: ChunkFilter) -> Result<Vec<ChunkMetadata>> {
        if let Some(file_hash) = filter.file_hash {
            let chunks = self.file_index.get_chunks(&file_hash).await?;
            Ok(chunks.into_iter().filter_map(|chunk| {
                chunk.id.map(|id| ChunkMetadata {
                    id,
                    file_hash: chunk.file_hash,
                    relative_path: chunk.relative_path,
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    has_embedding: chunk.embedding.is_some(),
                })
            }).collect())
        } else {
            // Would need to implement a global query in FileIndex
            unimplemented!("Global chunk listing not yet implemented")
        }
    }

    async fn get_file_chunks(&self, file_hash: FileHash) -> Result<Vec<Chunk>> {
        let chunk_refs = self.file_index.get_chunks(&file_hash).await?;
        Ok(chunk_refs.into_iter().map(Self::chunk_ref_to_chunk).collect())
    }
}

#[async_trait]
impl EmbeddingStore for SqliteStore {
    async fn store_embeddings(&self, chunk_ids: Vec<ChunkId>, embeddings: Vec<Vec<f32>>) -> Result<()> {
        if chunk_ids.len() != embeddings.len() {
            return Err(anyhow::anyhow!("Chunk IDs and embeddings count mismatch"));
        }
        
        // This would need to be implemented as an update operation in FileIndex
        // For now, this is a placeholder
        unimplemented!("Updating embeddings by chunk ID not yet implemented")
    }

    async fn search_similar(&self, _query: Vec<f32>, _limit: usize, _threshold: Option<f32>) -> Result<Vec<(ChunkId, f32)>> {
        // This would implement in-memory cosine similarity search across all embeddings
        unimplemented!("Vector similarity search not yet implemented")
    }

    async fn delete_embeddings(&self, _chunk_ids: Vec<ChunkId>) -> Result<()> {
        // Would set embedding to NULL for specified chunk IDs
        unimplemented!("Selective embedding deletion not yet implemented")
    }

    async fn get_embedding(&self, _chunk_id: ChunkId) -> Result<Option<Vec<f32>>> {
        // Would query embedding for specific chunk ID
        unimplemented!("Get embedding by chunk ID not yet implemented")
    }
}

#[async_trait]
impl CombinedStore for SqliteStore {
    async fn search_chunks(&self, query: Vec<f32>, limit: usize, threshold: Option<f32>) -> Result<Vec<(Chunk, f32)>> {
        // Get similar chunk IDs
        let similar_chunks = self.search_similar(query, limit, threshold).await?;
        
        // Fetch full chunk data
        let mut results = Vec::new();
        for (chunk_id, score) in similar_chunks {
            if let Some(chunk) = self.get_chunk(chunk_id).await? {
                results.push((chunk, score));
            }
        }
        
        Ok(results)
    }
}