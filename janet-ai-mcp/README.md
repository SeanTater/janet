# janet-ai-mcp

A **Model Context Protocol (MCP) Server** that provides semantic and regex search capabilities across codebases. Built for integration with AI assistants and code analysis tools, janet-ai-mcp leverages the Janet AI ecosystem to deliver powerful code search functionality through a standardized MCP interface.

## Overview

janet-ai-mcp serves as a bridge between AI assistants and the Janet AI codebase analysis tools, providing four core search commands through the MCP protocol:

- **`status`**: Database statistics and provider information
- **`regex_search`**: Pattern-based file content search with intelligent filtering
- **`semantic_search`**: Embedding-based similarity search for natural language queries
- **`delegate_search`**: Advanced search combining embeddings with LLM validation (planned)

## Features

### 🔍 **Multi-Modal Search**
- **Regex Search**: Fast pattern matching across project files with gitignore support
- **Semantic Search**: Vector similarity search using f16 embeddings for natural language queries
- **Configurable Filtering**: File type filters, dependency inclusion, documentation inclusion

### 🗄️ **Real Database Integration**
- **SQLite Storage**: Leverages janet-ai-retriever's robust chunk database
- **Embedding Management**: Integrates with janet-ai-embed for vector operations
- **Statistics Reporting**: Real-time database status and provider health information

### ⚡ **Performance & Reliability**
- **Async/Await**: Full async support for non-blocking operations
- **Graceful Fallbacks**: Semantic search degrades gracefully when embeddings unavailable
- **Resource Limits**: Configurable result limits to prevent overwhelming output
- **Error Handling**: Comprehensive error reporting with user-friendly messages

### 🔧 **MCP Protocol Compliance**
- **Standard Interface**: Full MCP v1.0 compatibility
- **Stdio Transport**: Standard input/output communication for AI assistant integration
- **Tool Schema**: Proper JSON schema definitions for all search parameters
- **Extensible Design**: Easy to add new search tools and capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Assistant (MCP Client)                    │
├─────────────────────────────────────────────────────────────────┤
│                    JSON-RPC over stdio                          │
├─────────────────────────────────────────────────────────────────┤
│                      janet-ai-mcp Server                        │
│  ┌───────────────┬──────────────────┬─────────────────────────┐ │
│  │ MCP Tools     │  Search Services │    Database Integration │ │
│  │               │                  │                         │ │
│  │ • status      │ • Regex Engine   │ • FileIndex (SQLite)    │ │
│  │ • regex_search│ • Cosine Sim     │ • ChunkRef Storage      │ │
│  │ • semantic_*  │ • File Walking   │ • Embedding Vectors     │ │
│  │ • delegate_*  │ • Pattern Match  │ • Metadata Queries      │ │
│  └───────────────┴──────────────────┴─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Janet AI Ecosystem                           │
│  ┌───────────────┬──────────────────┬─────────────────────────┐ │
│  │janet-ai-embed │janet-ai-retriever│  janet-ai-context       │ │
│  │               │                  │                         │ │
│  │ • FastEmbed   │ • FileIndex      │ • Text Chunking         │ │
│  │ • Providers   │ • SQLite Store   │ • Metadata Tracking     │ │
│  │ • f16 Vectors │ • Async DB Ops   │ • Content Parsing       │ │
│  └───────────────┴──────────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Installation & Usage

### Building from Source

```bash
# Build the MCP server
cargo build --release -p janet-ai-mcp

# Run with default settings (semantic search enabled)
./target/release/janet-ai-mcp

# Run with custom configuration
./target/release/janet-ai-mcp --root /path/to/codebase --no-semantic

# Enable experimental delegate search
./target/release/janet-ai-mcp --enable-delegate
```

### MCP Client Integration

Configure your MCP client (Claude Desktop, etc.) to use janet-ai-mcp:

```json
{
  "mcpServers": {
    "janet-ai-mcp": {
      "command": "/path/to/janet-ai-mcp",
      "args": ["--root", "/path/to/your/codebase"]
    }
  }
}
```

### Example Usage

Once connected through an MCP client, you can use these commands:

```javascript
// Check database status and provider health
status()

// Search for function definitions
regex_search({
  "pattern": "fn\\s+\\w+\\s*\\(",
  "file_type": "rs",
  "include_deps": false
})

// Find similar code using natural language
semantic_search({
  "query": "error handling and logging",
  "limit": 5,
  "threshold": 0.75
})

// Advanced search with LLM validation (when implemented)
delegate_search({
  "embedding_query": "database connection",
  "llm_query": "show me patterns for connection pooling",
  "candidates": 10
})
```

## Current Implementation Status

### ✅ **Completed Features**

#### Status Tool
- **Real Database Integration**: Connects to janet-ai-retriever SQLite database
- **Statistics Reporting**: Shows chunk counts, embedding status, file counts
- **Provider Health**: Reports embedding provider initialization status
- **Database Location**: Displays index database path and configuration

#### Regex Search Tool
- **File System Walking**: Uses `ignore` crate for gitignore-aware traversal
- **Pattern Matching**: Full regex support with proper error handling
- **Filtering Options**: File type, dependencies, documentation inclusion
- **Result Limiting**: Configurable result limits to prevent output overflow
- **Performance Optimized**: Parallel file processing with early termination

#### Semantic Search Tool
- **Embedding Integration**: Full integration with janet-ai-embed FastEmbedProvider
- **Vector Similarity**: Custom cosine similarity implementation for f16 vectors
- **Threshold Filtering**: Configurable similarity thresholds
- **Database Queries**: Efficient retrieval of chunks with embeddings
- **Error Handling**: Graceful fallback when embeddings unavailable

#### Delegate Search Tool (Framework)
- **Semantic Foundation**: Builds on semantic search for candidate retrieval
- **LLM Integration Points**: Architecture ready for LLM validation layer
- **Enhanced Results**: Currently shows enriched semantic search results
- **Future-Ready**: Placeholder implementation with clear integration path

### 🔄 **In Progress / Planned Features**

#### LLM Validation for Delegate Search
- **Integration Target**: Local LLM via llama.cpp or similar
- **Validation Pipeline**: Send candidates to LLM for relevance scoring
- **Result Reranking**: Combine semantic similarity with LLM feedback
- **Query Refinement**: Use separate queries for embedding and LLM phases

#### Enhanced Vector Operations
- **Optimized Similarity**: SIMD-accelerated cosine similarity calculations
- **Batch Processing**: Vectorized operations for multiple query comparisons
- **Index Structures**: Vector indexing for large-scale semantic search
- **Approximate Search**: LSH or similar for sub-linear search times

#### Real-time File Monitoring
- **Live Updates**: Integration with janet-ai-retriever's file watching
- **Incremental Indexing**: Update chunks and embeddings on file changes
- **Change Notifications**: MCP events for file system updates
- **Conflict Resolution**: Handle concurrent file modifications gracefully

### 🚧 **Integration Dependencies**

#### janet-ai-embed Completion
- **Current Status**: Provides functional FastEmbedProvider with model download
- **Needed**: Production-ready embedding generation with error recovery
- **Blocking**: Full semantic search requires stable embedding provider

#### janet-ai-retriever Indexing Pipeline
- **Current Status**: FileIndex provides database operations
- **Needed**: Automated indexing of codebases with embedding generation
- **Recommendation**: Run janet-ai-retriever CLI to populate database before using MCP server

#### Local LLM Integration
- **Current Status**: Framework ready, no implementation
- **Options**: llama.cpp, candle-transformers, or external API
- **Design Goal**: Local-first processing without external dependencies

## Configuration

### Command Line Options

```bash
janet-ai-mcp [OPTIONS]

OPTIONS:
    -r, --root <DIR>         Root directory to search [default: current directory]
        --no-semantic        Disable semantic search functionality
        --enable-delegate    Enable delegate search with LLM validation
    -h, --help              Print help information
    -V, --version           Print version information
```

### Environment Variables

```bash
# Override default model cache directory
JANET_MODEL_CACHE=/path/to/model/cache

# Configure embedding model
JANET_EMBED_MODEL=snowflake-arctic-embed-xs

# Database location override (for testing)
JANET_DB_PATH=/path/to/custom/database
```

### Runtime Configuration

The server adapts to available resources:

- **No Database**: Status reports empty database, searches return helpful messages
- **No Embeddings**: Semantic search gracefully disabled, regex search remains functional
- **Model Download Failure**: Falls back to regex-only mode with clear error reporting
- **Limited Memory**: Automatically adjusts batch sizes and result limits

## Development & Testing

### Running Tests

```bash
# Run all tests
cargo test -p janet-ai-mcp

# Test with database integration (requires setup)
cargo test -p janet-ai-mcp --features integration-tests

# Test individual components
cargo test -p janet-ai-mcp regex_search
cargo test -p janet-ai-mcp semantic_search
```

### Manual Testing

```bash
# Test server initialization
cargo run -p janet-ai-mcp -- --help

# Test with minimal setup (no semantic search)
echo '{"method":"status","params":{}}' | cargo run -p janet-ai-mcp -- --no-semantic

# Test regex search
echo '{"method":"regex_search","params":{"pattern":"fn\\s+main"}}' | \
  cargo run -p janet-ai-mcp
```

### Integration with Other Crates

```bash
# First, populate the database
cargo run -p janet-ai-retriever -- index /path/to/codebase

# Then run the MCP server
cargo run -p janet-ai-mcp -- --root /path/to/codebase
```

## Troubleshooting

### Common Issues

#### "No chunks with embeddings found"
- **Cause**: Database not populated or embeddings not generated
- **Solution**: Run `janet-ai-retriever index` to populate database
- **Workaround**: Use regex search which doesn't require embeddings

#### "Embedding provider not available"
- **Cause**: Model download failed or insufficient disk space
- **Solution**: Check network connection and disk space, restart server
- **Workaround**: Use `--no-semantic` flag to disable embedding features

#### "Invalid regex pattern"
- **Cause**: Malformed regular expression in search query
- **Solution**: Escape special characters, use raw strings for complex patterns
- **Example**: Use `\\\\` instead of `\\` for literal backslashes

#### "Database file not found"
- **Cause**: FileIndex database doesn't exist in expected location
- **Solution**: Check that `--root` points to indexed directory
- **Debug**: Look for `.janet.db` in root directory

### Performance Optimization

#### Large Codebases
- **Use File Type Filters**: Limit search to relevant extensions
- **Adjust Result Limits**: Reduce `limit` parameter for faster responses
- **Exclude Dependencies**: Set `include_deps: false` for project-only search
- **Monitor Memory**: Watch for OOM with very large embedding sets

#### Slow Semantic Search
- **Check Index Size**: Large embedding sets require more memory/time
- **Increase Threshold**: Higher thresholds return fewer, more relevant results
- **Batch Optimization**: Ensure janet-ai-embed uses efficient batch sizes
- **Consider Hardware**: Semantic search benefits from faster CPU/memory

## Contributing

### Adding New Search Tools

1. **Define Request Schema**: Add new struct in `server.rs` with `schemars` attributes
2. **Implement Tool Method**: Add async method with `#[tool]` attribute
3. **Integration**: Use existing `file_index` and `embedding_provider` fields
4. **Testing**: Add unit tests and integration tests
5. **Documentation**: Update this README with new tool capabilities

### Extending Search Capabilities

- **New Similarity Metrics**: Implement alternatives to cosine similarity
- **Advanced Filtering**: Add language-specific or domain-specific filters
- **Result Ranking**: Combine multiple signals for better result ordering
- **Caching**: Add query result caching for repeated searches

### Integration Guidelines

When integrating with new janet-ai-* crates:

1. **Dependency Management**: Add to `Cargo.toml` with proper version constraints
2. **Error Handling**: Implement graceful fallbacks for missing dependencies
3. **Configuration**: Add command-line flags for new features
4. **Documentation**: Update architecture diagrams and status reporting

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Related Projects

- **[janet-ai-retriever](../janet-ai-retriever)**: Core database and indexing functionality
- **[janet-ai-embed](../janet-ai-embed)**: Embedding generation and vector operations
- **[janet-ai-context](../janet-ai-context)**: Text chunking and content analysis
- **[MCP Specification](https://spec.modelcontextprotocol.io/)**: Protocol documentation
