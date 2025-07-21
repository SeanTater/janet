use clap::{Parser, Subcommand};
use half::f16;
use janet_ai_retriever::{
    retrieval::file_index::FileIndex,
    storage::{ChunkFilter, ChunkStore, CombinedStore, sqlite_store::SqliteStore},
};
use serde::Serialize;
use std::path::PathBuf;
use std::process;

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

            match format {
                OutputFormat::Json => {
                    #[derive(Serialize)]
                    struct StatusOutput {
                        index_statistics: janet_ai_retriever::status::IndexStatistics,
                        indexing_status: janet_ai_retriever::status::IndexingStatus,
                        index_health: janet_ai_retriever::status::IndexHealth,
                        indexing_configuration: janet_ai_retriever::status::IndexingConfiguration,
                        embedding_model_info:
                            Option<janet_ai_retriever::status::EmbeddingModelInfo>,
                        supported_file_types: Vec<String>,
                    }

                    let output = StatusOutput {
                        index_statistics: index_stats,
                        indexing_status,
                        index_health,
                        indexing_configuration: indexing_config,
                        embedding_model_info: model_info,
                        supported_file_types: supported_types,
                    };

                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                OutputFormat::Summary | OutputFormat::Full => {
                    println!("Janet AI Retriever Status");
                    println!("==========================");

                    println!("\n📊 Index Statistics:");
                    println!("  Total files: {}", index_stats.total_files);
                    println!("  Total chunks: {}", index_stats.total_chunks);
                    println!("  Total embeddings: {}", index_stats.total_embeddings);
                    println!("  Models count: {}", index_stats.models_count);
                    if let Some(db_size) = index_stats.database_size_bytes {
                        println!("  Database size: {db_size} bytes");
                    }

                    println!("\n⚙️  Indexing Status:");
                    println!(
                        "  Is running: {}",
                        if indexing_status.is_running {
                            "Yes"
                        } else {
                            "No"
                        }
                    );
                    println!("  Queue size: {}", indexing_status.queue_size);
                    println!("  Files processed: {}", indexing_status.files_processed);
                    println!("  Chunks created: {}", indexing_status.chunks_created);
                    println!(
                        "  Embeddings generated: {}",
                        indexing_status.embeddings_generated
                    );
                    println!("  Errors: {}", indexing_status.error_count);

                    println!("\n💚 Health Status:");
                    println!("  Overall status: {:?}", index_health.overall_status);
                    println!(
                        "  Database connected: {}",
                        if index_health.database_connected {
                            "Yes"
                        } else {
                            "No"
                        }
                    );
                    println!(
                        "  Database integrity: {}",
                        if index_health.database_integrity_ok {
                            "OK"
                        } else {
                            "Issues found"
                        }
                    );
                    if let Some(ref error) = index_health.database_error {
                        println!("  Database error: {error}");
                    }

                    println!("\n🔧 Configuration:");
                    println!("  Repository: {}", indexing_config.repository);
                    println!("  Base path: {}", indexing_config.base_path);
                    println!("  Indexing mode: {}", indexing_config.indexing_mode);
                    println!("  Max chunk size: {}", indexing_config.max_chunk_size);
                    println!("  Worker threads: {}", indexing_config.worker_thread_count);

                    if let Some(ref model) = model_info {
                        println!("\n🤖 Embedding Model:");
                        println!("  Model name: {}", model.model_name);
                        println!("  Provider: {}", model.provider);
                        println!("  Dimensions: {}", model.dimensions);
                        println!(
                            "  Normalized: {}",
                            if model.normalized { "Yes" } else { "No" }
                        );
                        println!("  Download status: {:?}", model.download_status);
                    } else {
                        println!("\n🤖 Embedding Model: None configured");
                    }

                    println!("\n📁 Supported File Types:");
                    let types_str = supported_types.join(", ");
                    println!("  {types_str}");
                }
            }

            Ok(())
        }
    }
}
