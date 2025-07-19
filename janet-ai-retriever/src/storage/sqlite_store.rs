use super::{
    Chunk, ChunkFilter, ChunkId, ChunkMetadata, ChunkStore, CombinedStore, EmbeddingStore, FileHash,
};
use crate::retrieval::file_index::{ChunkRef, FileIndex};
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
                if file_chunk.line_start == chunk_ref.line_start
                    && file_chunk.line_end == chunk_ref.line_end
                {
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
        let chunk_ref = self.file_index.get_chunk_by_id(id).await?;
        Ok(chunk_ref.map(Self::chunk_ref_to_chunk))
    }

    async fn update_chunk(&self, id: ChunkId, chunk: Chunk) -> Result<()> {
        let chunk_ref = Self::chunk_to_chunk_ref(chunk);
        self.file_index.update_chunk_by_id(id, &chunk_ref).await
    }

    async fn delete_chunks(&self, file_hash: FileHash) -> Result<usize> {
        self.file_index.delete_chunks(&file_hash).await
    }

    async fn list_chunks(&self, filter: ChunkFilter) -> Result<Vec<ChunkMetadata>> {
        if let Some(file_hash) = filter.file_hash {
            let chunks = self.file_index.get_chunks(&file_hash).await?;
            Ok(chunks
                .into_iter()
                .filter_map(|chunk| {
                    chunk.id.map(|id| ChunkMetadata {
                        id,
                        file_hash: chunk.file_hash,
                        relative_path: chunk.relative_path,
                        line_start: chunk.line_start,
                        line_end: chunk.line_end,
                        has_embedding: chunk.embedding.is_some(),
                    })
                })
                .collect())
        } else {
            // Would need to implement a global query in FileIndex
            unimplemented!("Global chunk listing not yet implemented")
        }
    }

    async fn get_file_chunks(&self, file_hash: FileHash) -> Result<Vec<Chunk>> {
        let chunk_refs = self.file_index.get_chunks(&file_hash).await?;
        Ok(chunk_refs
            .into_iter()
            .map(Self::chunk_ref_to_chunk)
            .collect())
    }
}

#[async_trait]
impl EmbeddingStore for SqliteStore {
    async fn store_embeddings(
        &self,
        chunk_ids: Vec<ChunkId>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        if chunk_ids.len() != embeddings.len() {
            return Err(anyhow::anyhow!("Chunk IDs and embeddings count mismatch"));
        }

        // Update each chunk's embedding
        for (chunk_id, embedding) in chunk_ids.into_iter().zip(embeddings.iter()) {
            self.file_index
                .update_chunk_embedding(chunk_id, Some(embedding))
                .await?;
        }

        Ok(())
    }

    async fn search_similar(
        &self,
        query: Vec<f32>,
        limit: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<(ChunkId, f32)>> {
        // Get all chunks with embeddings
        let chunks = self.file_index.get_all_chunks_with_embeddings().await?;

        // Calculate cosine similarity for each chunk
        let mut similarities: Vec<(ChunkId, f32)> = Vec::new();

        for chunk in chunks {
            if let (Some(id), Some(embedding)) = (chunk.id, chunk.embedding) {
                let similarity = cosine_similarity(&query, &embedding);

                // Apply threshold if provided
                if let Some(min_threshold) = threshold {
                    if similarity >= min_threshold {
                        similarities.push((id, similarity));
                    }
                } else {
                    similarities.push((id, similarity));
                }
            }
        }

        // Sort by similarity score (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply limit
        similarities.truncate(limit);

        Ok(similarities)
    }

    async fn delete_embeddings(&self, chunk_ids: Vec<ChunkId>) -> Result<()> {
        self.file_index.delete_embeddings_by_ids(&chunk_ids).await
    }

    async fn get_embedding(&self, chunk_id: ChunkId) -> Result<Option<Vec<f32>>> {
        let chunk = self.file_index.get_chunk_by_id(chunk_id).await?;
        Ok(chunk.and_then(|c| c.embedding))
    }
}

// Helper function for cosine similarity calculation
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[async_trait]
impl CombinedStore for SqliteStore {
    async fn search_chunks(
        &self,
        query: Vec<f32>,
        limit: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<(Chunk, f32)>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::file_index::FileRef;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_embedding_operations() -> anyhow::Result<()> {
        // Create test database
        let temp_dir = tempdir()?;
        let file_index = FileIndex::open_memory(temp_dir.path()).await?;
        let store = SqliteStore::new(file_index);

        // Insert a test file
        let file_ref = FileRef {
            relative_path: "test.rs".to_string(),
            content: b"test content".to_vec(),
            hash: [1; 32],
        };
        store.file_index.upsert_file(&file_ref).await?;

        // Insert test chunks
        let chunks = vec![
            Chunk {
                id: None,
                file_hash: [1; 32],
                relative_path: "test.rs".to_string(),
                line_start: 1,
                line_end: 5,
                content: "fn main() {}".to_string(),
                embedding: None,
            },
            Chunk {
                id: None,
                file_hash: [1; 32],
                relative_path: "test.rs".to_string(),
                line_start: 6,
                line_end: 10,
                content: "fn test() {}".to_string(),
                embedding: None,
            },
        ];

        let chunk_ids = store.insert_chunks(chunks).await?;
        assert_eq!(chunk_ids.len(), 2);

        // Store embeddings
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        store
            .store_embeddings(chunk_ids.clone(), embeddings.clone())
            .await?;

        // Verify embeddings were stored
        let embedding1 = store.get_embedding(chunk_ids[0]).await?;
        assert_eq!(embedding1, Some(vec![0.1, 0.2, 0.3]));

        // Test similarity search
        let query = vec![0.15, 0.25, 0.35];
        let results = store.search_similar(query, 10, None).await?;
        assert_eq!(results.len(), 2);
        // First result should be the more similar one
        assert_eq!(results[0].0, chunk_ids[0]);
        assert!(results[0].1 > results[1].1);

        // Test with threshold
        // First, let's check what the actual similarities are
        let query_for_threshold = vec![0.15, 0.25, 0.35];
        let all_results = store
            .search_similar(query_for_threshold.clone(), 10, None)
            .await?;

        // Use a threshold that will filter out at least one result
        let threshold = (all_results[0].1 + all_results[1].1) / 2.0; // midpoint between top two
        let results_with_threshold = store
            .search_similar(query_for_threshold, 10, Some(threshold))
            .await?;
        assert_eq!(results_with_threshold.len(), 1); // Only one should pass threshold

        // Test delete embeddings
        store.delete_embeddings(vec![chunk_ids[0]]).await?;
        let embedding_after_delete = store.get_embedding(chunk_ids[0]).await?;
        assert_eq!(embedding_after_delete, None);

        Ok(())
    }

    #[test]
    fn test_cosine_similarity() {
        // Test identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);

        // Test orthogonal vectors
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Test opposite vectors
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), -1.0);

        // Test normalized vectors
        let a = vec![0.6, 0.8];
        let b = vec![0.8, 0.6];
        let similarity = cosine_similarity(&a, &b);
        assert!((similarity - 0.96).abs() < 0.01);

        // Test zero vectors
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Test different length vectors
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
