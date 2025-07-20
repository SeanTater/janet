# Janet

A high-performance code retrieval and analysis system built in Rust, designed for building intelligent code search and understanding tools.

## Overview

Janet is a Rust workspace containing two complementary crates for semantic code analysis:

- **janet-context**: A library for chunking code/text into structured passages optimized for retrieval models (RAG systems)
- **janet-retriever**: A concurrent embedding database and file indexing system for semantic code search

## Features

### janet-context
- 🧩 **Smart Text Chunking**: Breaks code into semantic chunks while preserving metadata
- 📝 **Configurable Delimiters**: Customizable regex patterns for different content types
- 🏷️ **Rich Metadata**: Each chunk includes repository, file path, and position information
- 🖥️ **CLI Tool**: Process files directly from command line with JSON output
- 🔄 **Content Reconstruction**: Guarantees original content can be rebuilt from chunks

### janet-retriever
- 🗄️ **SQLite-based Storage**: Robust, concurrent database with WAL mode for multi-process safety
- 🔍 **Embedding Support**: Built-in storage for vector embeddings with efficient serialization
- ⚡ **Async/Await**: Full async support for non-blocking operations
- 🎯 **File Watching**: Real-time monitoring of code changes with debouncing
- 🏗️ **Modular Architecture**: Clean traits for storage backends and analysis engines
- 🧪 **Comprehensive Testing**: Memory-based testing with 100% async coverage
- 🖥️ **Interactive CLI**: Manage chunk databases with search, statistics, and data inspection

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
use janet_context::text::{TextContextBuilder, DEFAULT_MARKDOWN_DELIMITERS};

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

### Quick Example: File Indexing

```rust
use janet_retriever::retrieval::file_index::{FileIndex, FileRef};

let index = FileIndex::open("./project").await?;

// Index a file
let file_ref = FileRef {
    relative_path: "src/main.rs".to_string(),
    content: source_code.into_bytes(),
    hash: blake3::hash(&source_code).into(),
};

index.upsert_file(&file_ref).await?;
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

#### Available Commands

- **`init`**: Initialize the chunk database in `.code-assistant/index.db`
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
├── janet-context/          # Text chunking library
│   ├── src/
│   │   ├── text.rs        # Core chunking logic
│   │   └── bin/           # CLI tool
│   └── README.md
├── janet-retriever/        # Indexing and storage
│   ├── src/
│   │   ├── retrieval/     # File analysis and chunking
│   │   ├── storage/       # Database abstractions
│   │   └── migrations/    # Database schema
│   └── README.md
├── CLAUDE.md              # Development guidance
└── ARCHITECTURE_REFERENCE.md  # Detailed architecture notes
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific crate tests
cargo test -p janet-context
cargo test -p janet-retriever

# Run with output
cargo test -- --nocapture
```

### Development Commands

See [CLAUDE.md](./CLAUDE.md) for comprehensive development commands and architecture guidance.

## Roadmap

- 🔮 **Vector Similarity Search**: In-memory cosine similarity for semantic search
- 🔌 **Embedding Providers**: Support for OpenAI, BGE, and local models
- 🚀 **Performance Optimization**: Batch operations and connection pooling
- 📊 **Metrics & Monitoring**: Comprehensive observability
- 🔗 **API Server**: REST/GraphQL interface for external integrations

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
