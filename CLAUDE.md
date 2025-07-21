# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build & Test
```bash
# Build all workspace members
cargo build

# Run all tests
cargo test

# Run specific tests with output
cargo test -p janet-ai-mcp test_name -- --nocapture

# Run clippy (required before commits)
cargo clippy --all-targets --all-features -- -D warnings

# Format code (auto-applied by pre-commit)
cargo fmt
```

### CI Debugging
```bash
# Use act to test CI locally (much faster than push-and-wait)
act -W .github/workflows/ci.yml -j test --matrix os:ubuntu-latest

# Test specific job only
act -W .github/workflows/ci.yml -j test
```

### Examples
```bash
# MCP server examples
cargo run -p janet-ai-mcp -- --root .
cargo run --example end_to_end_indexing
cargo run --example working_demo
```

## Architecture

This is a Rust workspace with four crates for semantic code search and AI integration:

### janet-ai-context
Text chunking library for RAG systems
- **TextContextBuilder**: Configurable text chunking with delimiters and size limits
- **CLI tool**: `janet-ai-context-cli` for processing files into JSON chunks

### janet-ai-embed
High-performance embedding generation library
- **FastEmbed Integration**: Multiple transformer models (BGE, E5, ModernBERT)
- **Async Processing**: Full async/await support with automatic model caching
- **Provider Pattern**: Extensible trait-based design

### janet-ai-retriever
Complete indexing and search system
- **IndexingEngine**: File discovery, chunking, and embedding generation
- **SQLite Storage**: Text search and vector similarity (cosine) search using f16 embeddings
- **File Watching**: Real-time monitoring with debouncing and graceful degradation
- **CLI Interface**: Database management and search tools

### janet-ai-mcp
MCP (Model Context Protocol) server for AI integration
- **Protocol Implementation**: Full MCP 2024-11-05 specification support
- **Semantic Search**: Tool for AI-powered code search across repositories
- **Status Endpoint**: Comprehensive system health and configuration reporting

## Key Technical Details

### Database Storage
- Uses `.janet.db` SQLite file directly in base directory (not `.code-assistant/index.db`)
- Avoids permission issues in CI environments by eliminating subdirectory creation
- Supports both persistent and in-memory databases for testing

### Development Philosophy
- **Local-first**: Prefer SQLite over Pinecone, Ollama over OpenAI APIs
- **Testing**: Use tracing (not log), focus on happy paths first, use "sans io" patterns
- **Performance**: f16 embeddings for memory efficiency, async-first architecture

## Workflows

### Git & CI
- **Never commit to main** - use feature branches and conventional commits
- **Pre-commit hooks**: Auto-format, clippy, trailing whitespace, conventional commit validation
- **CI Debugging**: Use `act` locally instead of push-and-wait cycles
- **Testing**: Role-play as QA engineer, consider edge cases and dataset sizes

### Common Issues & Solutions

#### CI Test Failures
- **Use act first**: `act -W .github/workflows/ci.yml -j test`
- **Database errors**: Check for `.janet.db` path references (not `.code-assistant/`)
- **Clippy failures**: Run `cargo clippy --all-targets --all-features -- -D warnings` locally

#### Integration Tests
- **Timeout issues**: Use `#[ignore]` for problematic tests, document in TODO.md
- **Permission errors**: Ensure using `.janet.db` directly, not subdirectories
- **MCP protocol**: Tests require process spawning and stdio communication

#### Performance
- **Large embeddings**: Use f16 instead of f32 for memory efficiency
- **File watching**: Debounce file events, graceful degradation on errors
- **Async coordination**: Task queues with priority and retry logic

## Project Status
- **janet-ai-context**: ✅ Production ready with CLI tools
- **janet-ai-embed**: ✅ Production ready with FastEmbed integration
- **janet-ai-retriever**: ✅ Feature complete with storage and search
- **janet-ai-mcp**: 🔄 Active development, integration tests require CI debugging

All crates have comprehensive test coverage and working examples.
