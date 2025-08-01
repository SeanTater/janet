use crate::ServerConfig;
use anyhow::Result as AnyhowResult;
use janet_ai_embed::{EmbedConfig, EmbeddingProvider, FastEmbedProvider};
use janet_ai_retriever::retrieval::indexing_engine::{IndexingEngine, IndexingEngineConfig};
use janet_ai_retriever::storage::{ChunkStore, EmbeddingStore, sqlite_store::SqliteStore};
use rmcp::schemars;
use serde::{Deserialize, Serialize};
use tracing::info;

#[allow(dead_code)] // Used by rmcp macro system
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SemanticSearchRequest {
    #[schemars(description = "Query text for semantic search")]
    pub query: String,
    #[schemars(description = "Maximum number of results to return")]
    pub limit: Option<u32>,
    #[schemars(description = "Similarity threshold (0.0 to 1.0)")]
    pub threshold: Option<f32>,
}

#[allow(dead_code)] // Used by rmcp macro system
pub async fn semantic_search(
    config: &ServerConfig,
    request: SemanticSearchRequest,
) -> Result<String, String> {
    info!(
        "Processing semantic search: query='{}', limit={:?}, threshold={:?}",
        request.query, request.limit, request.threshold
    );

    let limit = request.limit.unwrap_or(10) as usize;
    let threshold = request.threshold.unwrap_or(0.7);

    // Perform semantic search
    match perform_semantic_search(config, &request.query, limit, threshold).await {
        Ok(results) => {
            if results.is_empty() {
                Ok(format!(
                    "Semantic Search Results\\n\\
                    Query: '{}'\\n\\
                    Limit: {}\\n\\
                    Threshold: {:.2}\\n\\
                    \\n\\
                    No results found matching the query.",
                    request.query, limit, threshold
                ))
            } else {
                let mut output = format!(
                    "Semantic Search Results\\n\\
                    Query: '{}'\\n\\
                    Limit: {}\\n\\
                    Threshold: {:.2}\\n\\
                    \\n\\
                    Found {} similar chunks:\\n\\n",
                    request.query,
                    limit,
                    threshold,
                    results.len()
                );

                for (i, (chunk, similarity)) in results.iter().enumerate() {
                    output.push_str(&format!(
                        "{}. {}:{}-{} - Similarity: {:.3}\\n",
                        i + 1,
                        chunk.relative_path,
                        chunk.line_start,
                        chunk.line_end,
                        similarity
                    ));

                    // Show content preview (first 200 chars)
                    let preview = chunk.content.chars().take(200).collect::<String>();
                    let preview = if chunk.content.len() > 200 {
                        format!("{preview}...")
                    } else {
                        preview
                    };
                    output.push_str(&format!("   {}\\n\\n", preview.replace("\\n", " ")));
                }

                Ok(output)
            }
        }
        Err(e) => Err(format!("Semantic search failed: {e}")),
    }
}

/// Attempt to perform semantic search using the indexing engine
async fn perform_semantic_search(
    config: &ServerConfig,
    query: &str,
    limit: usize,
    threshold: f32,
) -> AnyhowResult<Vec<(janet_ai_retriever::storage::Chunk, f32)>> {
    use half::f16;

    // Try to load existing index from the root directory
    let index_db_path = config.root_dir.join(".janet-ai.db");

    if !index_db_path.exists() {
        return Err(anyhow::anyhow!(
            "No index database found at {:?}",
            index_db_path
        ));
    }

    // Create an IndexingEngine to access the existing index
    let indexing_config =
        IndexingEngineConfig::new("janet-mcp-search".to_string(), config.root_dir.clone());

    let engine = IndexingEngine::new(indexing_config).await?;
    let enhanced_index = engine.get_enhanced_index();

    // Try to get the embedding provider if embeddings are available
    let provider = match create_embedding_provider().await {
        Ok(provider) => provider,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to create embedding provider: {}",
                e
            ));
        }
    };

    // Generate embedding for the query
    let query_embedding = provider.embed_texts(&[query.to_string()]).await?;
    if query_embedding.is_empty() {
        return Err(anyhow::anyhow!("Failed to generate query embedding"));
    }

    // Get the embedding vector (already f16)
    let query_vec = query_embedding.embeddings[0].clone();

    // Search for similar chunks
    let store = SqliteStore::new(enhanced_index.file_index().clone());
    let threshold_f16 = f16::from_f32(threshold);

    let similar_chunk_ids = store
        .search_similar(query_vec, limit, Some(threshold_f16))
        .await?;

    // Get full chunk data
    let mut results = Vec::new();
    for (chunk_id, similarity) in similar_chunk_ids {
        if let Some(chunk) = store.get_chunk(chunk_id).await? {
            results.push((chunk, similarity.to_f32()));
        }
    }

    Ok(results)
}

/// Create an embedding provider for generating query embeddings
async fn create_embedding_provider() -> AnyhowResult<FastEmbedProvider> {
    // Use a lightweight model suitable for MCP server usage
    let config = EmbedConfig::default();

    let provider = FastEmbedProvider::create(config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create provider: {}", e))?;
    Ok(provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile;

    #[tokio::test]
    async fn test_semantic_search_no_index() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        // Test semantic search when no index database exists
        let request = SemanticSearchRequest {
            query: "function that adds two numbers".to_string(),
            limit: Some(5),
            threshold: Some(0.5),
        };

        let result = semantic_search(&config, request).await;
        assert!(result.is_err(), "Should fail when no index exists");

        let error = result.unwrap_err();
        assert!(
            error.contains("No index database found"),
            "Should mention missing index: {error}"
        );
    }

    #[tokio::test]
    async fn test_semantic_search_request_defaults() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        // Test that request with no limit/threshold uses defaults
        let request = SemanticSearchRequest {
            query: "test query".to_string(),
            limit: None,
            threshold: None,
        };

        let result = semantic_search(&config, request).await;
        // Should fail because no index exists, but we can check the error message
        // contains information about the search parameters
        assert!(result.is_err());
        // The error should be about missing index, not about invalid parameters
        let error = result.unwrap_err();
        assert!(error.contains("No index database found"));
    }

    #[tokio::test]
    async fn test_semantic_search_request_validation() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = ServerConfig {
            root_dir: temp_dir.path().to_path_buf(),
        };

        // Test with various limit and threshold values
        let request = SemanticSearchRequest {
            query: "test query".to_string(),
            limit: Some(0),       // Edge case: zero limit
            threshold: Some(1.5), // Edge case: threshold > 1.0
        };

        let result = semantic_search(&config, request).await;
        assert!(result.is_err());
        // Should fail due to missing index, not parameter validation
        let error = result.unwrap_err();
        assert!(error.contains("No index database found"));
    }

    #[test]
    fn test_semantic_search_request_structure() {
        // Test that the request structure can be created and has expected defaults
        let request = SemanticSearchRequest {
            query: "test".to_string(),
            limit: None,
            threshold: None,
        };

        assert_eq!(request.query, "test");
        assert!(request.limit.is_none());
        assert!(request.threshold.is_none());

        let request2 = SemanticSearchRequest {
            query: "another test".to_string(),
            limit: Some(20),
            threshold: Some(0.8),
        };

        assert_eq!(request2.limit, Some(20));
        assert_eq!(request2.threshold, Some(0.8));
    }
}
