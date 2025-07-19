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
cargo build -p janet-retriever
cargo build -p janet-context
```

### Test
```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p janet-retriever
cargo test -p janet-context

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

## High-Level Architecture

This is a Rust workspace containing two related crates for code retrieval and analysis:

### janet-context
A library for chunking text/code into structured passages for retrieval models (RAG systems). Key features:
- **TextContextBuilder**: Configures how text is chunked with customizable delimiters and max chunk sizes
- **TextChunk**: Represents a text segment with metadata (repo, path, sequence)
- **CLI tool**: `janet-context-cli` for processing files into JSON chunks
- Focuses on preserving metadata and creating retrieval-optimized text segments

### janet-retriever
A retrieval system with file indexing and analysis capabilities. Currently under development with:
- **retrieval module**: Contains analyzer, directory_watcher, and file_index submodules
- **File monitoring**: Uses notify for real-time file system watching
- **Storage**: Integrates fjall (key-value store) and lancedb (vector database)
- **Embeddings**: Planned BERT integration for semantic search (BertChunkConfig)

### Architecture Inspiration
The project references ChunkHound architecture (see ARCHITECTURE_REFERENCE.md), which is a sophisticated MCP Server for semantic code search. Key patterns to follow:
- Protocol-first design for AI integration
- Async-first architecture with proper task coordination
- Provider pattern for extensible embeddings/databases
- Real-time file monitoring with graceful degradation
- Comprehensive error handling

## Current Status
- janet-context: Functional library with CLI tool
- janet-retriever: Basic structure in place, main components need implementation
- Build warnings exist for unused fields in BertChunkConfig (analyzer.rs:21-24)