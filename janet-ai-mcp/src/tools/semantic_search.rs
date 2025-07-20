use crate::ServerConfig;
use anyhow::Result as AnyhowResult;
use janet_ai_embed::{EmbedConfig, EmbeddingProvider, FastEmbedProvider};
use janet_ai_retriever::retrieval::{
    indexing_engine::{IndexingEngine, IndexingEngineConfig},
    indexing_mode::IndexingMode,
};
use janet_ai_retriever::storage::{ChunkStore, EmbeddingStore, sqlite_store::SqliteStore};
use rmcp::schemars;
use serde::Deserialize;
use tempfile;
use tracing::{info, warn};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SemanticSearchRequest {
    #[schemars(description = "Query text for semantic search")]
    pub query: String,
    #[schemars(description = "Maximum number of results to return")]
    pub limit: Option<u32>,
    #[schemars(description = "Similarity threshold (0.0 to 1.0)")]
    pub threshold: Option<f32>,
}

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

    // Try to perform real semantic search
    match perform_real_semantic_search(config, &request.query, limit, threshold).await {
        Ok(results) => {
            if results.is_empty() {
                Ok(format!(
                    "Semantic Search Results\\n\\
                    Query: '{}'\\n\\
                    Limit: {}\\n\\
                    Threshold: {:.2}\\n\\
                    \\n\\
                    No results found. This could mean:\\n\\
                    1. The query doesn't match any indexed content\\n\\
                    2. The repository hasn't been indexed yet\\n\\
                    3. No embeddings have been generated\\n\\
                    \\n\\
                    Try running 'janet-ai-retriever index' first to populate the database.",
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
        Err(e) => {
            warn!("Real semantic search failed: {}", e);
            // Fall back to mock results
            let mock_results = generate_mock_semantic_results(&request.query, limit, threshold);
            Ok(format!(
                "Semantic Search Results (Mock)\\n\\
                Query: '{}'\\n\\
                Limit: {}\\n\\
                Threshold: {:.2}\\n\\
                \\n\\
                Found {} similar chunks:\\n\\
                \\n\\
                {}\\n\\
                \\n\\
                Note: Using mock implementation. Real semantic search failed: {}\\n\\
                To use real semantic search:\\n\\
                1. Run 'janet-ai-retriever index <directory>' to populate the database\\n\\
                2. Ensure embedding models are available\\n\\
                3. Check that the indexed database exists",
                request.query,
                limit,
                threshold,
                mock_results.len(),
                mock_results.join("\\n\\n"),
                e
            ))
        }
    }
}

/// Attempt to perform real semantic search using the indexing engine
async fn perform_real_semantic_search(
    config: &ServerConfig,
    query: &str,
    limit: usize,
    threshold: f32,
) -> AnyhowResult<Vec<(janet_ai_retriever::storage::Chunk, f32)>> {
    use half::f16;

    // Try to load existing index from the root directory
    let index_db_path = config.root_dir.join(".janet").join("index.db");

    if !index_db_path.exists() {
        return Err(anyhow::anyhow!(
            "No index database found at {:?}",
            index_db_path
        ));
    }

    // Create an IndexingEngine to access the existing index
    let indexing_config =
        IndexingEngineConfig::new("janet-mcp-search".to_string(), config.root_dir.clone())
            .with_mode(IndexingMode::ReadOnly);

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
    let temp_dir =
        tempfile::tempdir().map_err(|e| anyhow::anyhow!("Failed to create temp dir: {}", e))?;
    let config = EmbedConfig::default_with_path(temp_dir.path())
        .with_batch_size(1)
        .with_normalize(true);

    let provider = FastEmbedProvider::create(config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create provider: {}", e))?;
    Ok(provider)
}

fn generate_mock_semantic_results(query: &str, limit: usize, _threshold: f32) -> Vec<String> {
    // Generate realistic mock results based on query content
    let results = if query.to_lowercase().contains("error") {
        vec![
            "1. src/lib.rs (lines 45-52) - Similarity: 0.89\\n\\
            pub fn handle_error(err: Error) -> Result<(), String> {\\n\\
                tracing::error!(\"Operation failed: {:?}\", err);\\n\\
                Err(format!(\"Error: {}\", err))\\n\\
            }",
            "2. src/server.rs (lines 156-163) - Similarity: 0.83\\n\\
            match self.process_request().await {\\n\\
                Ok(response) => Ok(response),\\n\\
                Err(e) => {\\n\\
                    error!(\"Request processing failed: {}\", e);\\n\\
                    Err(e)\\n\\
                }\\n\\
            }",
            "3. src/tools/regex_search.rs (lines 78-82) - Similarity: 0.78\\n\\
            Err(e) => {\\n\\
                warn!(\"Failed to read file {:?}: {}\", path, e);\\n\\
                continue;\\n\\
            }",
        ]
    } else if query.to_lowercase().contains("config") {
        vec![
            "1. src/lib.rs (lines 16-28) - Similarity: 0.92\\n\\
            #[derive(Debug, Clone)]\\n\\
            pub struct ServerConfig {\\n\\
                pub root_dir: PathBuf,\\n\\
            }\\n\\
            \\n\\
            impl Default for ServerConfig {",
            "2. src/main.rs (lines 25-30) - Similarity: 0.86\\n\\
            let mut config = ServerConfig::default();\\n\\
            if let Some(root_dir) = matches.get_one::<PathBuf>(\"root\") {\\n\\
                config.root_dir = root_dir.clone();\\n\\
            }",
            "3. src/server.rs (lines 51-54) - Similarity: 0.81\\n\\
            pub async fn new(config: ServerConfig) -> Result<Self> {\\n\\
                info!(\"Initializing Janet MCP server with root: {:?}\", config.root_dir);\\n\\
                Ok(Self { config })\\n\\
            }",
        ]
    } else if query.to_lowercase().contains("search") {
        vec![
            "1. src/tools/regex_search.rs (lines 15-22) - Similarity: 0.94\\n\\
            pub async fn regex_search(config: &ServerConfig, request: RegexSearchRequest) -> Result<String, String> {\\n\\
                info!(\"Processing regex search: pattern='{}', globs={:?}\", request.pattern, request.globs);\\n\\
                let regex = match Regex::new(&request.pattern) {",
            "2. src/tools/semantic_search.rs (lines 8-13) - Similarity: 0.88\\n\\
            pub struct SemanticSearchRequest {\\n\\
                pub query: String,\\n\\
                pub limit: Option<u32>,\\n\\
                pub threshold: Option<f32>,\\n\\
            }",
            "3. src/server.rs (lines 75-80) - Similarity: 0.82\\n\\
            #[tool(description = \"Search files using regex patterns\")]\\n\\
            async fn regex_search(&self, request: RegexSearchRequest) -> Result<String, String> {\\n\\
                tools::regex_search::regex_search(&self.config, request).await\\n\\
            }",
        ]
    } else {
        vec![
            "1. src/lib.rs (lines 32-38) - Similarity: 0.75\\n\\
            pub async fn run_server(config: ServerConfig) -> Result<()> {\\n\\
                info!(\"Starting Janet MCP server\");\\n\\
                let janet_server = JanetMcpServer::new(config).await?;\\n\\
                janet_server.serve_stdio().await\\n\\
            }",
            "2. src/server.rs (lines 102-108) - Similarity: 0.71\\n\\
            pub async fn serve_stdio(&self) -> Result<()> {\\n\\
                info!(\"Starting MCP server with stdio transport\");\\n\\
                let transport = (stdin(), stdout());\\n\\
                let server = self.clone().serve(transport).await?;\\n\\
            }",
        ]
    };

    results
        .into_iter()
        .take(limit)
        .map(|s| s.to_string())
        .collect()
}
