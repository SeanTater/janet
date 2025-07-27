//! Retrieval subsystem for janet-ai-retriever
//!
//! This module provides the core functionality for indexing, chunking, and retrieving
//! code files for semantic search. The subsystem is designed around a layered architecture:
//!
//! ## Core Components
//!
//! - **FileIndex**: Low-level SQLite database operations for files and chunks
//! - **EnhancedFileIndex**: High-level wrapper with metadata tracking and model compatibility
//! - **IndexingEngine**: Orchestrates the complete indexing pipeline
//! - **DirectoryWatcher**: Monitors filesystem changes for incremental updates
//! - **TaskQueue**: Manages async work distribution across workers
//!
//! ## Supporting Components
//!
//! - **Analyzer**: Determines file types and processing strategies
//! - **ChunkingStrategy**: Configures how files are split into searchable chunks
//!
//! ## Architecture Flow
//!
//! ```text
//! Files → Analyzer → ChunkingStrategy → FileIndex/EnhancedFileIndex
//!   ↑                                         ↓
//! DirectoryWatcher ← IndexingEngine → TaskQueue → Embeddings
//! ```
//!
//! ## Usage Examples
//!
//! Basic indexing:

pub mod analyzer;
pub mod chunking_strategy;
pub mod enhanced_index;
pub mod file_index;
pub mod indexing_engine;
pub mod task_queue;
