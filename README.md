# Janet

A high-performance code retrieval and analysis system built in Rust, designed for building intelligent code search and understanding tools.

## Overview

Janet is a Rust workspace containing four complementary crates for AI-powered code analysis and retrieval:

- **[janet-ai-context](./janet-ai-context/README.md)**: A library for chunking code/text into structured passages optimized for retrieval models (RAG systems)
- **[janet-ai-retriever](./janet-ai-retriever/README.md)**: A complete indexing system with file monitoring, text search, and vector similarity search
- **[janet-ai-mcp](./janet-ai-mcp/README.md)**: MCP (Model Context Protocol) server for AI integration and lightweight search tools
- **[janet-ai-gtk](./janet-ai-gtk/README.md)**: Modern GTK4 desktop application with chat interface for code search

## Features

### janet-ai-context
- 🧩 **Smart Text Chunking**: Breaks code into semantic chunks while preserving metadata
- 📝 **Configurable Delimiters**: Customizable regex patterns for different content types
- 🏷️ **Rich Metadata**: Each chunk includes repository, file path, and position information
- 🖥️ **CLI Tool**: Process files directly from command line with JSON output
- 🔄 **Content Reconstruction**: Guarantees original content can be rebuilt from chunks

### janet-ai-retriever
- 🔄 **Indexing Engine**: Orchestrates file discovery and chunking
- 🗄️ **SQLite Storage**: Robust database with text search
- 🔍 **Text Search**: Substring and text-based search (semantic/vector search removed)
- 🎯 **File Watching**: Real-time monitoring of code changes with debouncing
- 📋 **Task Queue**: Priority-based background processing with retry logic
- ⚡ **Async Architecture**: Full async support for non-blocking operations
- 🖥️ **Complete CLI**: Manage databases with search, statistics, and data inspection
- 📊 **Rich Examples**: Comprehensive examples showing end-to-end workflows

### janet-ai-mcp
- 🔌 **MCP Server**: Full Model Context Protocol implementation for AI integration
- 🔍 **Search Tools**: Regex and semantic search capabilities for AI assistants
- 📊 **Status Reporting**: Comprehensive system health and configuration monitoring
- ⚡ **Async Processing**: Non-blocking search operations with proper error handling
- 🛠️ **Development Tools**: CLI tools for testing and debugging MCP functionality

### janet-ai-gtk
- 🖥️ **Modern Desktop UI**: Clean GTK4 interface with contemporary design
- 💬 **Chat Interface**: Intuitive conversational interface for code exploration
- 🎯 **Dual Search Modes**: Both semantic and regex search with IRC-style commands
- 📁 **Repository Selection**: Easy folder selection or command-line specification
- ⚡ **Real-time Results**: Fast search with immediate response display
- 🎨 **Syntax Highlighting**: Formatted code display with monospace styling

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

### Quick Example: End-to-End Indexing

```rust
use janet_ai_retriever::retrieval::indexing_engine::{IndexingEngine, IndexingEngineConfig};

// Set up indexing engine
let config = IndexingEngineConfig::new("my-project".to_string(), project_path)
    .with_chunk_size(500);

let mut engine = IndexingEngine::new_memory(config).await?;
// Start with full reindex
engine.start(true).await?;

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

# Search chunks (text-based)
cargo run -p janet-ai-retriever -- list

# Show database statistics
cargo run -p janet-ai-retriever -- stats

# Get help for any command
cargo run -p janet-ai-retriever -- --help
```

### GTK Desktop Application

For a user-friendly graphical interface:

```bash
# Install required system dependencies (Ubuntu/Debian)
sudo apt install libgtk-4-dev

# Build and run the GTK app
cargo run -p janet-ai-gtk

# Or with a specific repository
cargo run -p janet-ai-gtk -- /path/to/your/repo
```

### MCP Server

Run the MCP server for AI integration:

```bash
# Start MCP server for current directory
cargo run -p janet-ai-mcp -- --root .

# Test MCP server functionality
cargo run -p janet-ai-mcp --example working_demo
```

### Examples

Run comprehensive examples showing full workflows:

```bash
# End-to-end indexing with text search
cargo run --example end_to_end_indexing

# Working demo with file indexing
cargo run --example working_demo

<!-- Embedding generation demos removed from main README. See `janet-ai-embed/` for archived examples. -->
```

#### Available Commands

- **`init`**: Initialize the chunk database in `.janet-ai.db`
- **`list`**: List chunks with optional filtering by file hash
- **`get <id>`**: Retrieve a specific chunk by its database ID
-- **`search`**: Find chunks using text-based matching (semantic/vector search removed)
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
├── janet-ai-embed/         # (archived) Embedding generation library (not part of runtime)
│   ├── src/
│   │   ├── provider.rs    # FastEmbed provider implementation
│   │   ├── config.rs      # Configuration and builders
│   │   └── downloader.rs  # Model downloading
│   ├── examples/          # Archived embedding examples
│   └── README.md
├── janet-ai-retriever/     # Indexing and storage
│   ├── src/
│   │   ├── retrieval/     # Indexing engine and file analysis
│   │   ├── storage/       # Database abstractions and search
│   │   └── main.rs        # CLI application
│   ├── examples/          # End-to-end workflow examples
│   ├── migrations/        # Database schema
│   └── README.md
├── janet-ai-mcp/           # MCP server for AI integration
│   ├── src/
│   │   ├── tools/         # Search tool implementations
│   │   ├── server.rs      # MCP protocol handling
│   │   └── main.rs        # Server application
│   └── README.md
├── janet-ai-gtk/           # GTK4 desktop application
│   ├── src/
│   │   ├── ui.rs          # Main UI components
│   │   ├── style.css      # GTK styling
│   │   └── main.rs        # Application entry point
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
