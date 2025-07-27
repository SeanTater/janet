use clap::{Parser, Subcommand};
use half::f16;
use janet_ai_embed::{EmbedConfig, TokenizerConfig};
use janet_ai_retriever::{
    retrieval::{
        file_index::FileIndex,
        indexing_engine::{IndexingEngine, IndexingEngineConfig},
        indexing_mode::IndexingMode,
    },
    storage::{ChunkFilter, ChunkStore, CombinedStore, sqlite_store::SqliteStore},
};
use serde::Serialize;
use std::path::PathBuf;
use std::process;
use tempfile::tempdir;
use tokio::time::Duration;

/// A CLI tool to interact with the janet-ai-retriever chunk database.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Base directory containing the .janet-ai.db database file
    #[arg(short, long, default_value = ".")]
    base_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Initialize the chunk database
    Init,
    /// Index files in a repository and create chunks with embeddings
    Index {
        /// Repository path to index (defaults to base-dir)
        #[arg(long)]
        repo: Option<PathBuf>,
        /// Maximum number of worker threads
        #[arg(long, default_value_t = 4)]
        max_workers: usize,
        /// Chunk size in characters
        #[arg(long, default_value_t = 1000)]
        chunk_size: usize,
        /// Embedding model to use for semantic search
        #[arg(long, default_value = "snowflake-arctic-embed-xs")]
        embedding_model: String,
        /// Force full reindex (ignore existing chunks)
        #[arg(long)]
        force: bool,
    },
    /// List chunks in the database
    List {
        /// Filter by file hash (hex encoded)
        #[arg(long)]
        file_hash: Option<String>,
        /// Limit number of results
        #[arg(short, long, default_value_t = 100)]
        limit: usize,
        /// Output format
        #[arg(short, long, default_value = "summary")]
        format: OutputFormat,
    },
    /// Get a specific chunk by ID
    Get {
        /// Chunk ID
        id: i64,
        /// Output format
        #[arg(short, long, default_value = "full")]
        format: OutputFormat,
    },
    /// Search for similar chunks using embedding similarity
    Search {
        /// Query embedding values (comma-separated floats)
        #[arg(long, value_delimiter = ',')]
        embedding: Vec<f32>,
        /// Maximum number of results
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
        /// Minimum similarity threshold (0.0 to 1.0)
        #[arg(short, long)]
        threshold: Option<f32>,
        /// Output format
        #[arg(short, long, default_value = "summary")]
        format: OutputFormat,
    },
    /// Show database statistics
    Stats,
    /// Show comprehensive status information
    Status {
        /// Output format
        #[arg(short, long, default_value = "summary")]
        format: OutputFormat,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum OutputFormat {
    Summary,
    Full,
    Json,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "summary" => Ok(OutputFormat::Summary),
            "full" => Ok(OutputFormat::Full),
            "json" => Ok(OutputFormat::Json),
            _ => Err(format!("Invalid format: {s}")),
        }
    }
}

#[derive(Serialize)]
struct ChunkOutput {
    id: i64,
    file_hash: String,
    relative_path: String,
    line_start: usize,
    line_end: usize,
    content: String,
    has_embedding: bool,
}

#[derive(Serialize)]
struct SimilarityResult {
    chunk: ChunkOutput,
    similarity: f32,
}

#[derive(Serialize)]
struct DatabaseStats {
    total_chunks: usize,
    chunks_with_embeddings: usize,
    unique_files: std::collections::HashSet<String>,
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Init => {
            let _file_index = FileIndex::open(&args.base_dir).await?;
            println!("Initialized chunk database at {}", args.base_dir.display());
            println!(
                "Database location: {}/.janet-ai.db",
                args.base_dir.display()
            );
            Ok(())
        }
        Commands::Index {
            repo,
            max_workers,
            chunk_size,
            embedding_model,
            force,
        } => {
            let repo_path = repo.unwrap_or_else(|| args.base_dir.clone());

            println!("🚀 Starting indexing process...");
            println!("   Repository: {}", repo_path.display());
            println!("   Database: {}/.janet-ai.db", args.base_dir.display());
            println!("   Max workers: {max_workers}");
            println!("   Chunk size: {chunk_size}");
            println!("   Embedding model: {embedding_model}");
            println!("   Force reindex: {force}");

            // Set up embedding configuration
            let embedding_temp_dir = tempdir()?;
            let model_dir = embedding_temp_dir.path().join(&embedding_model);
            let tokenizer_config = TokenizerConfig::standard(&model_dir);
            let embed_config = EmbedConfig::new(
                embedding_temp_dir.path(),
                &embedding_model,
                tokenizer_config,
            )
            .with_batch_size(8)
            .with_normalize(true);

            // Set up the indexing engine configuration
            let indexing_config =
                IndexingEngineConfig::new("cli-indexing".to_string(), repo_path.clone())
                    .with_mode(if force {
                        IndexingMode::FullReindex
                    } else {
                        IndexingMode::ContinuousMonitoring
                    })
                    .with_max_workers(max_workers)
                    .with_chunk_size(chunk_size)
                    .with_embedding_config(embed_config);

            println!("⚙️  Initializing IndexingEngine...");

            // Create indexing engine using the specified base directory
            let mut engine = IndexingEngine::new(indexing_config).await?;

            println!("✅ IndexingEngine initialized successfully");

            // Start the engine and perform indexing
            println!("🔄 Starting indexing...");
            engine.start().await?;

            // Wait for indexing to complete
            let mut attempts = 0;
            let max_attempts = 60; // 60 seconds max wait

            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;

                // Process any pending tasks
                engine.process_pending_tasks().await?;

                let queue_size = engine.get_queue_size().await;
                let stats = engine.get_stats().await;

                if attempts % 5 == 0 || queue_size == 0 {
                    println!(
                        "📊 Queue size: {}, Files processed: {}, Chunks created: {}, Embeddings: {}",
                        queue_size,
                        stats.files_processed,
                        stats.chunks_created,
                        stats.embeddings_generated
                    );
                }

                if queue_size == 0 && stats.files_processed > 0 {
                    println!("✅ Indexing completed!");
                    break;
                }

                attempts += 1;
                if attempts >= max_attempts {
                    println!(
                        "⚠️  Timeout waiting for indexing to complete (queue_size: {queue_size})"
                    );
                    println!("   You may want to wait longer or check for errors");
                    break;
                }
            }

            // Get final statistics
            let final_stats = engine.get_stats().await;
            let index_stats = engine.get_index_stats().await?;

            println!("\n📈 Final Statistics:");
            println!("   Files processed: {}", final_stats.files_processed);
            println!("   Chunks created: {}", final_stats.chunks_created);
            println!(
                "   Embeddings generated: {}",
                final_stats.embeddings_generated
            );
            println!("   Errors encountered: {}", final_stats.errors);
            println!("   Total files in index: {}", index_stats.files_count);
            println!("   Total chunks in index: {}", index_stats.chunks_count);
            println!("   Embedding models: {}", index_stats.models_count);

            // Clean up
            engine.shutdown().await?;

            println!("\n🎉 Indexing completed successfully!");
            println!("   You can now use 'janet-ai-retriever search' to find similar chunks");
            println!("   Or use the janet-ai-mcp server for semantic search over MCP");

            Ok(())
        }
        Commands::List {
            file_hash,
            limit,
            format,
        } => {
            let file_index = FileIndex::open(&args.base_dir).await?;
            let store = SqliteStore::new(file_index);

            let filter = if let Some(hash_str) = file_hash {
                let hash_bytes = hex::decode(&hash_str)
                    .map_err(|_| anyhow::anyhow!("Invalid hex hash: {}", hash_str))?;
                if hash_bytes.len() != 32 {
                    return Err(anyhow::anyhow!("Hash must be 32 bytes (64 hex characters)"));
                }
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&hash_bytes);
                ChunkFilter {
                    file_hash: Some(hash),
                    path_prefix: None,
                    has_embedding: None,
                }
            } else {
                ChunkFilter {
                    file_hash: None,
                    path_prefix: None,
                    has_embedding: None,
                }
            };

            let mut chunk_metadata = store.list_chunks(filter).await?;
            chunk_metadata.truncate(limit);

            match format {
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&chunk_metadata)?);
                }
                OutputFormat::Summary => {
                    println!("Found {} chunks:", chunk_metadata.len());
                    for metadata in chunk_metadata {
                        println!(
                            "  ID: {} | File: {} | Lines: {}-{} | Embedding: {}",
                            metadata.id,
                            metadata.relative_path,
                            metadata.line_start,
                            metadata.line_end,
                            if metadata.has_embedding { "✓" } else { "✗" }
                        );
                    }
                }
                OutputFormat::Full => {
                    for metadata in chunk_metadata {
                        if let Some(chunk) = store.get_chunk(metadata.id).await? {
                            let output = ChunkOutput {
                                id: chunk.id.unwrap_or(0),
                                file_hash: hex::encode(chunk.file_hash),
                                relative_path: chunk.relative_path,
                                line_start: chunk.line_start,
                                line_end: chunk.line_end,
                                content: chunk.content,
                                has_embedding: chunk.embedding.is_some(),
                            };
                            println!("Chunk ID: {}", output.id);
                            println!("File: {}", output.relative_path);
                            println!("Lines: {}-{}", output.line_start, output.line_end);
                            println!("File Hash: {}", output.file_hash);
                            println!(
                                "Has Embedding: {}",
                                if output.has_embedding { "Yes" } else { "No" }
                            );
                            println!("Content:\n{}", output.content);
                            println!("---");
                        }
                    }
                }
            }
            Ok(())
        }
        Commands::Get { id, format } => {
            let file_index = FileIndex::open(&args.base_dir).await?;
            let store = SqliteStore::new(file_index);

            if let Some(chunk) = store.get_chunk(id).await? {
                let output = ChunkOutput {
                    id: chunk.id.unwrap_or(0),
                    file_hash: hex::encode(chunk.file_hash),
                    relative_path: chunk.relative_path,
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    content: chunk.content,
                    has_embedding: chunk.embedding.is_some(),
                };

                match format {
                    OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    }
                    OutputFormat::Summary => {
                        println!("Chunk ID: {}", output.id);
                        println!("File: {}", output.relative_path);
                        println!("Lines: {}-{}", output.line_start, output.line_end);
                        println!(
                            "Has Embedding: {}",
                            if output.has_embedding { "Yes" } else { "No" }
                        );
                        println!(
                            "Content preview: {}",
                            output.content.chars().take(100).collect::<String>()
                        );
                    }
                    OutputFormat::Full => {
                        println!("Chunk ID: {}", output.id);
                        println!("File: {}", output.relative_path);
                        println!("Lines: {}-{}", output.line_start, output.line_end);
                        println!("File Hash: {}", output.file_hash);
                        println!(
                            "Has Embedding: {}",
                            if output.has_embedding { "Yes" } else { "No" }
                        );
                        println!("Content:\n{}", output.content);
                    }
                }
            } else {
                println!("Chunk with ID {id} not found");
            }
            Ok(())
        }
        Commands::Search {
            embedding,
            limit,
            threshold,
            format,
        } => {
            if embedding.is_empty() {
                return Err(anyhow::anyhow!("Embedding vector cannot be empty"));
            }

            // Validate threshold range
            if let Some(thresh) = threshold {
                if !(0.0..=1.0).contains(&thresh) {
                    return Err(anyhow::anyhow!(
                        "Similarity threshold must be between 0.0 and 1.0, got {thresh}"
                    ));
                }
            }

            let file_index = FileIndex::open(&args.base_dir).await?;
            let store = SqliteStore::new(file_index);

            // Convert f32 embedding to f16
            let embedding_f16: Vec<f16> = embedding.iter().map(|&x| f16::from_f32(x)).collect();
            let threshold_f16 = threshold.map(f16::from_f32);

            let results = store
                .search_chunks(embedding_f16, limit, threshold_f16)
                .await?;

            match format {
                OutputFormat::Json => {
                    let similarity_results: Vec<SimilarityResult> = results
                        .into_iter()
                        .map(|(chunk, similarity)| SimilarityResult {
                            chunk: ChunkOutput {
                                id: chunk.id.unwrap_or(0),
                                file_hash: hex::encode(chunk.file_hash),
                                relative_path: chunk.relative_path,
                                line_start: chunk.line_start,
                                line_end: chunk.line_end,
                                content: chunk.content,
                                has_embedding: chunk.embedding.is_some(),
                            },
                            similarity: similarity.to_f32(),
                        })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&similarity_results)?);
                }
                OutputFormat::Summary => {
                    println!("Found {} similar chunks:", results.len());
                    for (chunk, similarity) in results {
                        println!(
                            "  Similarity: {:.3} | ID: {} | File: {} | Lines: {}-{}",
                            similarity.to_f32(),
                            chunk.id.unwrap_or(0),
                            chunk.relative_path,
                            chunk.line_start,
                            chunk.line_end
                        );
                    }
                }
                OutputFormat::Full => {
                    for (chunk, similarity) in results {
                        println!("Similarity: {:.3}", similarity.to_f32());
                        println!("Chunk ID: {}", chunk.id.unwrap_or(0));
                        println!("File: {}", chunk.relative_path);
                        println!("Lines: {}-{}", chunk.line_start, chunk.line_end);
                        println!("File Hash: {}", hex::encode(chunk.file_hash));
                        println!("Content:\n{}", chunk.content);
                        println!("---");
                    }
                }
            }
            Ok(())
        }
        Commands::Stats => {
            let file_index = FileIndex::open(&args.base_dir).await?;
            let store = SqliteStore::new(file_index);

            // Get all chunks
            let all_chunks = store
                .list_chunks(ChunkFilter {
                    file_hash: None,
                    path_prefix: None,
                    has_embedding: None,
                })
                .await?;
            let chunks_with_embeddings = all_chunks.iter().filter(|c| c.has_embedding).count();
            let unique_files: std::collections::HashSet<String> =
                all_chunks.iter().map(|c| c.relative_path.clone()).collect();

            let stats = DatabaseStats {
                total_chunks: all_chunks.len(),
                chunks_with_embeddings,
                unique_files: unique_files.clone(),
            };

            println!("Database Statistics:");
            println!("  Total chunks: {}", stats.total_chunks);
            println!("  Chunks with embeddings: {}", stats.chunks_with_embeddings);
            println!("  Unique files: {}", stats.unique_files.len());

            if !stats.unique_files.is_empty() {
                println!("  Files:");
                for file in stats.unique_files.iter().take(10) {
                    println!("    {file}");
                }
                if stats.unique_files.len() > 10 {
                    println!("    ... and {} more", stats.unique_files.len() - 10);
                }
            }

            Ok(())
        }
        Commands::Status { format } => {
            use janet_ai_retriever::{
                retrieval::{
                    indexing_engine::{IndexingEngine, IndexingEngineConfig},
                    indexing_mode::IndexingMode,
                },
                status::StatusApi,
            };

            // Create a read-only indexing engine for status queries
            let config = IndexingEngineConfig::new("cli-repo".to_string(), args.base_dir.clone())
                .with_mode(IndexingMode::ReadOnly);

            let engine = IndexingEngine::new(config.clone()).await?;
            let enhanced_index = engine.get_enhanced_index();

            // Get all status information
            let index_stats = StatusApi::get_index_statistics(enhanced_index).await?;
            let indexing_status = StatusApi::get_indexing_status(&engine).await?;
            let index_health = StatusApi::get_index_health(enhanced_index).await?;
            let indexing_config = StatusApi::get_indexing_config(&config).await?;
            let model_info = StatusApi::get_embedding_model_info(None).await?;
            let supported_types = StatusApi::get_supported_file_types(&config).await?;
            let database_info =
                StatusApi::get_database_info(enhanced_index, &args.base_dir).await?;
            let dependency_versions = StatusApi::get_dependency_versions().await?;
            let consistency_report = StatusApi::validate_index_consistency(enhanced_index).await?;
            let file_system_status = StatusApi::get_file_system_status(&config).await?;

            #[derive(Serialize)]
            struct StatusOutput {
                index_statistics: janet_ai_retriever::status::IndexStatistics,
                indexing_status: janet_ai_retriever::status::IndexingStatus,
                index_health: janet_ai_retriever::status::IndexHealth,
                indexing_configuration: janet_ai_retriever::status::IndexingConfiguration,
                embedding_model_info: Option<janet_ai_retriever::status::EmbeddingModelInfo>,
                supported_file_types: Vec<String>,
                database_info: janet_ai_retriever::status::DatabaseInfo,
                dependency_versions: janet_ai_retriever::status::DependencyVersions,
                consistency_report: janet_ai_retriever::status::IndexConsistencyReport,
                file_system_status: janet_ai_retriever::status::FileSystemStatus,
            }

            let output = StatusOutput {
                index_statistics: index_stats,
                indexing_status,
                index_health,
                indexing_configuration: indexing_config,
                embedding_model_info: model_info,
                supported_file_types: supported_types,
                database_info,
                dependency_versions,
                consistency_report,
                file_system_status,
            };

            match format {
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                OutputFormat::Summary | OutputFormat::Full => {
                    println!("{}", toml::to_string_pretty(&output)?);
                }
            }

            Ok(())
        }
    }
}
