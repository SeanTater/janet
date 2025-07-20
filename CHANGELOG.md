# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with Rust workspace
- `janet-context` crate for intelligent text chunking
  - Smart text segmentation with configurable regex delimiters
  - Rich metadata preservation (repo, path, sequence, content)
  - CLI tool for processing files with JSON output
  - Content reconstruction guarantees
  - Default Markdown delimiter patterns
- `janet-retriever` crate for code indexing and storage
  - SQLite-based storage with WAL mode for concurrency safety
  - Full async/await support with tokio
  - Embedding storage as BLOBs with efficient serialization
  - File indexing with Blake3 hashing
  - Chunk storage with line-level precision
  - Multi-process safe database operations
  - Comprehensive test suite with memory database support
- Storage abstraction traits (`ChunkStore`, `EmbeddingStore`, `CombinedStore`)
- File watching infrastructure with debouncing
- Database migration system
- Comprehensive documentation and development guides

### Technical Details
- Rust 2024 Edition with latest stable features
- SQLite with WAL journaling for concurrent access
- ACID transactions with proper error handling
- Memory-based testing infrastructure
- Modular trait-based architecture
- Real-time file monitoring with notify crate
- Blake3 hashing for content integrity

### Development Infrastructure
- MIT License
- Comprehensive README with examples
- CLAUDE.md development guidance
- Architecture reference documentation
- Full test coverage with async testing
- Cargo workspace configuration
- Git repository with proper branching strategy

## [0.1.0] - 2025-01-19

### Added
- Initial release with core functionality
- Text chunking and code indexing capabilities
- SQLite-based concurrent storage system
- CLI tools for text processing
- Development documentation and examples
