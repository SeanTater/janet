//! janet-ai-retriever: Semantic code search and indexing system
//!
//! This crate provides a complete solution for indexing and searching source code files
//! using semantic embeddings and traditional text search. It's designed to work with
//! large codebases and provides real-time indexing capabilities.
//!
//! ## Key Modules
//!
//! - **[`retrieval`]**: Core indexing engine, file analysis, and chunking strategies
//! - **[`storage`]**: Storage abstraction layer with SQLite implementation
//! - **[`status`]**: System diagnostics and health monitoring APIs
//!
//! ## Quick Start
//!
//!
//! ## Architecture
//!
//! ```text
//! Files → Analyzer → Chunking → Embeddings → SQLite Storage
//!   ↑                                           ↓
//! FileWatcher ← IndexingEngine ← TaskQueue ← Search APIs
//! ```

pub mod retrieval;
pub mod status;
pub mod storage;
