# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Build
```bash
# Build all workspace members
cargo build

# Build with release optimizations
cargo build --release

# Build a specific crate
cargo build -p janet-ai-retriever
cargo build -p janet-ai-context
cargo build -p janet-ai-embed
```

### Test
```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p janet-ai-retriever
cargo test -p janet-ai-context
cargo test -p janet-ai-embed

# Run a specific test by name
cargo test test_name

# Run tests with output displayed
cargo test -- --nocapture
```

### Lint
```bash
# Run clippy linter on all crates
cargo clippy

# Apply clippy suggestions automatically
cargo clippy --fix

# Run clippy with stricter warnings
cargo clippy -- -W clippy::all
```

### Format
```bash
# Check formatting
cargo fmt -- --check

# Apply formatting
cargo fmt
```

### Examples
```bash
# Run end-to-end indexing demo with text search
cargo run --example end_to_end_indexing

# Run working demo with file indexing
cargo run --example working_demo

# Run embedding generation demo
cargo run --example embedding_demo

# Run simple embedding example
cargo run -p janet-ai-embed --example simple_embedding

# Run ModernBERT embedding example
cargo run -p janet-ai-embed --example modernbert_example
```

## High-Level Architecture

This is a Rust workspace containing three related crates for code retrieval and AI-powered analysis:

### janet-ai-context
A library for chunking text/code into structured passages for retrieval models (RAG systems). Key features:
- **TextContextBuilder**: Configures how text is chunked with customizable delimiters and max chunk sizes
- **TextChunk**: Represents a text segment with metadata (repo, path, sequence)
- **CLI tool**: `janet-ai-context-cli` for processing files into JSON chunks
- Focuses on preserving metadata and creating retrieval-optimized text segments

### janet-ai-embed
A comprehensive embedding library with multiple provider support. Key features:
- **FastEmbed Integration**: High-performance embedding generation using FastEmbed
- **Model Management**: Automatic model downloading and caching
- **Multiple Models**: Support for BGE, E5, ModernBERT, and other transformer models
- **Async Processing**: Full async/await support for embedding generation
- **Configuration Builder**: Flexible configuration with derive_builder pattern
- **Provider Abstraction**: Clean trait-based design for different embedding backends

### janet-ai-retriever
A complete retrieval system with file indexing, text search, and embedding capabilities. Key features:
- **IndexingEngine**: Orchestrates file discovery, chunking, and embedding generation
- **SQLite Storage**: Robust database with text search and vector similarity search
- **File Watching**: Real-time monitoring with notify and debouncing
- **Task Queue**: Priority-based background processing with retry logic
- **Text Search**: Substring search with SQL LIKE queries and proper escaping
- **Vector Search**: Cosine similarity search over f16 embeddings
- **CLI Interface**: Complete command-line tool for database management
- **Examples**: Comprehensive examples showing end-to-end workflows

### Architecture Inspiration
The project references ChunkHound architecture (see ARCHITECTURE_REFERENCE.md), which is a sophisticated MCP Server for semantic code search. Key patterns to follow:
- Protocol-first design for AI integration
- Async-first architecture with proper task coordination
- Provider pattern for extensible embeddings/databases
- Real-time file monitoring with graceful degradation
- Comprehensive error handling

## Current Status
- **janet-ai-context**: Fully functional library with CLI tool and comprehensive chunking
- **janet-ai-embed**: Production-ready embedding library with FastEmbed integration
- **janet-ai-retriever**: Feature-complete indexing system with storage, search, and CLI
- All crates have comprehensive test coverage and working examples

## Development Philosophy
- Use tracing rather than log
- Use "sans io" (separate io operations from pure functions) when possible to keep things easy to test
- Lean into a local-first policy:
  - Prefer sqlite over pinecone
  - Prefer ollama/llamacpp over openai/anthropic
  - Prefer bundled solutions over separate when feasible (like a rust library rather than a python dependency or separate API)
- Document concisely, but completely
- Clear code is better than clear docs
- Write benchmarks and keep track of them, but don't obsess

## Git Workflow
- Never commit directly to main. Work in feature branches, and create PRs.
- Commit and push (to a feature branch and PR) whenever you feel the code is generally functional and tests are green.

## PR and Testing Guidelines
- Before pushing a new PR, try role playing as a QA engineer first. Be skeptical that the tests cover all the use cases. Consider edge cases, large and small datasets, etc.
- Focus on happy paths in tests before sad paths.
