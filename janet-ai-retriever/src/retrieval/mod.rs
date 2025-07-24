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
//! - **IndexingMode**: Controls indexing behavior (full reindex, incremental, etc.)
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
//! ```rust,no_run
//! use janet_ai_retriever::retrieval::{
//!     file_index::FileIndex,
//!     indexing_engine::{IndexingEngine, IndexingEngineConfig},
//!     indexing_mode::IndexingMode,
//! };
//! use std::path::Path;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create basic file index
//! let file_index = FileIndex::open(Path::new(".")).await?;
//!
//! // Or use the full indexing engine
//! let config = IndexingEngineConfig::new("my-project".to_string(), Path::new(".").to_path_buf())
//!     .with_mode(IndexingMode::ContinuousMonitoring);
//! let mut engine = IndexingEngine::new(config).await?;
//! engine.start().await?;
//! # Ok(())
//! # }
//! ```

pub mod analyzer;
pub mod chunking_strategy;
pub mod enhanced_index;
pub mod file_index;
pub mod indexing_engine;
pub mod indexing_mode;
pub mod task_queue;
