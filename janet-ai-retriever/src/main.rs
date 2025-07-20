use clap::{Parser, Subcommand};
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
    /// Base directory containing the .code-assistant database
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
                "Database location: {}/.code-assistant/index.db",
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

            let results = store.search_chunks(embedding, limit, threshold).await?;

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
                            similarity,
                        })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&similarity_results)?);
                }
                OutputFormat::Summary => {
                    println!("Found {} similar chunks:", results.len());
                    for (chunk, similarity) in results {
                        println!(
                            "  Similarity: {:.3} | ID: {} | File: {} | Lines: {}-{}",
                            similarity,
                            chunk.id.unwrap_or(0),
                            chunk.relative_path,
                            chunk.line_start,
                            chunk.line_end
                        );
                    }
                }
                OutputFormat::Full => {
                    for (chunk, similarity) in results {
                        println!("Similarity: {similarity:.3}");
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
    }
}
