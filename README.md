# Janet

A high-performance code retrieval and analysis system built in Rust, designed for building intelligent code search and understanding tools.

## Overview

Janet is a Rust workspace containing three complementary crates for AI-powered code analysis and retrieval:

- **janet-ai-context**: A library for chunking code/text into structured passages optimized for retrieval models (RAG systems)
- **janet-ai-embed**: A comprehensive embedding library with FastEmbed integration and multiple model support
- **janet-ai-retriever**: A complete indexing system with file monitoring, text search, and vector similarity search

## Features

### janet-ai-context
- 🧩 **Smart Text Chunking**: Breaks code into semantic chunks while preserving metadata
- 📝 **Configurable Delimiters**: Customizable regex patterns for different content types
- 🏷️ **Rich Metadata**: Each chunk includes repository, file path, and position information
- 🖥️ **CLI Tool**: Process files directly from command line with JSON output
- 🔄 **Content Reconstruction**: Guarantees original content can be rebuilt from chunks

### janet-ai-embed
- 🤖 **FastEmbed Integration**: High-performance embedding generation using FastEmbed
- 📦 **Automatic Model Management**: Downloads and caches models automatically
- 🎯 **Multiple Model Support**: BGE, E5, ModernBERT, and other transformer models
- ⚡ **Async Processing**: Full async/await support for embedding generation
- 🔧 **Builder Pattern**: Flexible configuration with derive_builder
- 🏗️ **Provider Abstraction**: Clean trait-based design for different backends

### janet-ai-retriever
- 🔄 **Indexing Engine**: Orchestrates file discovery, chunking, and embedding generation
- 🗄️ **SQLite Storage**: Robust database with text search and vector similarity search
- 🔍 **Dual Search**: Both substring text search and cosine similarity vector search
- 🎯 **File Watching**: Real-time monitoring of code changes with debouncing
- 📋 **Task Queue**: Priority-based background processing with retry logic
- ⚡ **Async Architecture**: Full async support for non-blocking operations
- 🖥️ **Complete CLI**: Manage databases with search, statistics, and data inspection
- 📊 **Rich Examples**: Comprehensive examples showing end-to-end workflows

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/SeanTater/janet.git
cd janet

# Build all crates
cargo build --release

# Run tests
cargo test
```

### Quick Example: Text Chunking

```rust
use janet_ai_context::text::{TextContextBuilder, DEFAULT_MARKDOWN_DELIMITERS};

let builder = TextContextBuilder::new(
    "my_project".to_string(),
    "src/main.rs".to_string(),
    &DEFAULT_MARKDOWN_DELIMITERS,
    500  // max chunk size
);

let chunks = builder.get_chunks("fn main() {\n    println!(\"Hello, world!\");\n}");
for chunk in chunks {
    println!("Chunk {}: {}", chunk.sequence, chunk.chunk_text);
}
```

### Quick Example: Embedding Generation

```rust
use janet_ai_embed::config::FastEmbedConfigBuilder;
use janet_ai_embed::provider::FastEmbedProvider;

// Configure and create embedding provider
let config = FastEmbedConfigBuilder::default()
    .model_name("BAAI/bge-small-en-v1.5")
    .build()?;

let provider = FastEmbedProvider::try_new(config).await?;

// Generate embeddings
let texts = vec!["Hello world", "Rust programming"];
let embeddings = provider.generate_embeddings(&texts).await?;
```

### Quick Example: End-to-End Indexing

```rust
use janet_ai_retriever::retrieval::indexing_engine::{IndexingEngine, IndexingEngineConfig};
use janet_ai_retriever::retrieval::indexing_mode::IndexingMode;

// Set up indexing engine
let config = IndexingEngineConfig::new("my-project".to_string(), project_path)
    .with_mode(IndexingMode::FullReindex)
    .with_chunk_size(500);

let mut engine = IndexingEngine::new_memory(config).await?;
engine.start().await?;

// Engine will discover, chunk, and optionally embed all files
```

## CLI Tools

### janet-ai-context-cli

Process text files into structured JSON chunks:

```bash
# Process a file
cargo run -p janet-ai-context --bin janet-ai-context-cli -- -i src/main.rs -r my_project -p src/main.rs

# Process from stdin
echo "fn main() {}" | cargo run -p janet-ai-context --bin janet-ai-context-cli -- -r my_project -p stdin
```

### janet-ai-retriever

Interact with the chunk database:

```bash
# Initialize a new database
cargo run -p janet-ai-retriever -- init

# List all chunks
cargo run -p janet-ai-retriever -- list

# Get detailed chunk information
cargo run -p janet-ai-retriever -- get 123

# Search similar chunks (if embeddings are available)
cargo run -p janet-ai-retriever -- search --embedding 0.1,0.2,0.3 --limit 5

# Show database statistics
cargo run -p janet-ai-retriever -- stats

# Get help for any command
cargo run -p janet-ai-retriever -- --help
```

### Examples

Run comprehensive examples showing full workflows:

```bash
# End-to-end indexing with text search
cargo run --example end_to_end_indexing

# Working demo with file indexing
cargo run --example working_demo

# Embedding generation demo
cargo run --example embedding_demo

# Simple embedding example
cargo run -p janet-ai-embed --example simple_embedding

# ModernBERT embedding example
cargo run -p janet-ai-embed --example modernbert_example
```

#### Available Commands

- **`init`**: Initialize the chunk database in `.janet.db`
- **`list`**: List chunks with optional filtering by file hash
- **`get <id>`**: Retrieve a specific chunk by its database ID
- **`search`**: Find similar chunks using cosine similarity on embeddings
- **`stats`**: Display database statistics including chunk counts and file coverage

#### Output Formats

Most commands support multiple output formats via the `--format` flag:
- **`summary`**: Human-readable overview (default)
- **`full`**: Complete chunk details including content
- **`json`**: Machine-readable JSON for integration

## Architecture

Janet follows clean architecture principles with clear separation of concerns:

```
┌─────────────────────────────────────────────┐
│                   CLI Tools                 │
├─────────────────────────────────────────────┤
│              Application Layer              │
│  ┌─────────────────┬─────────────────────┐  │
│  │  janet-context  │   janet-retriever   │  │
│  │  (Text Chunking)│   (Indexing/Search) │  │
│  └─────────────────┴─────────────────────┘  │
├─────────────────────────────────────────────┤
│               Storage Layer                 │
│  ┌─────────────────┬─────────────────────┐  │
│  │     SQLite      │      Embeddings     │  │
│  │  (Metadata/CRUD)│    (Vector Store)   │  │
│  └─────────────────┴─────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Development

### Requirements
- Rust 2024 Edition (latest stable)
- SQLite 3.x
- Optional: Docker for containerized development

### Project Structure
```
janet/
├── janet-ai-context/       # Text chunking library
│   ├── src/
│   │   ├── text.rs        # Core chunking logic
│   │   └── bin/           # CLI tool
│   └── README.md
├── janet-ai-embed/         # Embedding generation library
│   ├── src/
│   │   ├── provider.rs    # FastEmbed provider implementation
│   │   ├── config.rs      # Configuration and builders
│   │   └── downloader.rs  # Model downloading
│   ├── examples/          # Embedding examples
│   └── README.md
├── janet-ai-retriever/     # Indexing and storage
│   ├── src/
│   │   ├── retrieval/     # Indexing engine and file analysis
│   │   ├── storage/       # Database abstractions and search
│   │   └── main.rs        # CLI application
│   ├── examples/          # End-to-end workflow examples
│   ├── migrations/        # Database schema
│   └── README.md
├── CLAUDE.md              # Development guidance
└── ARCHITECTURE_REFERENCE.md  # Detailed architecture notes
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific crate tests
cargo test -p janet-ai-context
cargo test -p janet-ai-embed
cargo test -p janet-ai-retriever

# Run with output
cargo test -- --nocapture
```

### Development Commands

See [CLAUDE.md](./CLAUDE.md) for comprehensive development commands and architecture guidance.

## Roadmap

- 🌐 **Additional Embedding Providers**: Support for OpenAI, Anthropic, and Cohere APIs
- 🚀 **Performance Optimization**: Batch operations and connection pooling
- 📊 **Metrics & Monitoring**: Comprehensive observability and performance tracking
- 🔗 **API Server**: REST/GraphQL interface for external integrations
- 🎯 **Advanced Search**: Hybrid search combining text and vector similarity
- 📈 **Scalability**: Support for larger codebases and distributed indexing

## Contributing

Contributions are welcome! Please see our development guidelines in [CLAUDE.md](./CLAUDE.md).

### Development Workflow
1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `cargo fmt && cargo clippy`
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Inspiration

This project draws architectural inspiration from [ChunkHound](ARCHITECTURE_REFERENCE.md), a sophisticated MCP server for semantic code search, adapting its patterns for Rust-based development tools.
