//! Indexing mode configuration for controlling indexing behavior.
//!
//! This module defines the operating modes for the indexing system, controlling
//! whether the system performs full reindexing, incremental monitoring, or read-only
//! operations. Each mode has different resource requirements and use cases.
//!
//! ## Indexing Modes
//!
//! ### FullReindex
//! Complete rebuild of the entire index from scratch.
//! - **Use case**: Initial setup, major model changes, corrupted index recovery
//! - **Resources**: High CPU, memory, and storage I/O during rebuild
//! - **Duration**: Proportional to codebase size (minutes to hours)
//! - **Safety**: Replaces all existing data
//!
//! ### ContinuousMonitoring (Default)
//! Incremental updates based on filesystem changes.
//! - **Use case**: Active development environments, production deployments
//! - **Resources**: Low baseline usage, spikes during file changes
//! - **Duration**: Near real-time updates
//! - **Safety**: Preserves existing data, only updates changed files
//!
//! ### ReadOnly
//! No indexing operations, only search queries.
//! - **Use case**: Search-only services, shared read-only environments
//! - **Resources**: Minimal (no embedding generation or file watching)
//! - **Duration**: Instant startup
//! - **Safety**: No data modifications
//!
//! ## Usage
//!
//! ```rust,no_run
//! use janet_ai_retriever::retrieval::indexing_mode::IndexingMode;
//!
//! // Configure for different scenarios
//! let initial_setup = IndexingMode::FullReindex;
//! let development = IndexingMode::ContinuousMonitoring;
//! let production_search = IndexingMode::ReadOnly;
//!
//! // Check capabilities
//! if development.allows_indexing() {
//!     println!("Will process file changes");
//! }
//!
//! if development.allows_file_watching() {
//!     println!("Will monitor filesystem");
//! }
//!
//! // Parse from string (useful for CLI/config)
//! let mode: IndexingMode = "continuous-monitoring".parse().unwrap();
//! ```
//!
//! ## Mode Selection Guidelines
//!
//! ### When to use FullReindex
//! - Setting up a new index
//! - Switching embedding models
//! - Recovering from database corruption
//! - Major codebase restructuring
//!
//! ### When to use ContinuousMonitoring
//! - Active development (default choice)
//! - Production with ongoing changes
//! - CI/CD pipelines that need fresh indexes
//!
//! ### When to use ReadOnly
//! - Search-only microservices
//! - Load-balanced search replicas
//! - Demo/presentation environments
//! - When filesystem watching is problematic

use serde::{Deserialize, Serialize};

/// Defines the operating mode for the file indexing system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexingMode {
    /// Perform a complete reindex of all files in the directory
    /// This will scan all files, rechunk them, regenerate embeddings, and update the database
    FullReindex,

    /// Continuously monitor for file changes and update the index incrementally
    /// New/modified files will be processed, deleted files will be removed from index
    ContinuousMonitoring,

    /// Read-only mode - no indexing operations will be performed
    /// The system can only serve search queries from existing index
    ReadOnly,
}

impl IndexingMode {
    /// Check if this mode allows indexing operations
    pub fn allows_indexing(&self) -> bool {
        matches!(
            self,
            IndexingMode::FullReindex | IndexingMode::ContinuousMonitoring
        )
    }

    /// Check if this mode allows file watching
    pub fn allows_file_watching(&self) -> bool {
        matches!(self, IndexingMode::ContinuousMonitoring)
    }

    /// Check if this mode is read-only
    pub fn is_read_only(&self) -> bool {
        matches!(self, IndexingMode::ReadOnly)
    }
}

impl Default for IndexingMode {
    fn default() -> Self {
        Self::ContinuousMonitoring
    }
}

impl std::fmt::Display for IndexingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexingMode::FullReindex => write!(f, "full-reindex"),
            IndexingMode::ContinuousMonitoring => write!(f, "continuous-monitoring"),
            IndexingMode::ReadOnly => write!(f, "read-only"),
        }
    }
}

impl std::str::FromStr for IndexingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full-reindex" | "full_reindex" | "full" => Ok(IndexingMode::FullReindex),
            "continuous-monitoring" | "continuous_monitoring" | "continuous" | "monitor" => {
                Ok(IndexingMode::ContinuousMonitoring)
            }
            "read-only" | "read_only" | "readonly" | "read" => Ok(IndexingMode::ReadOnly),
            _ => Err(format!(
                "Invalid indexing mode: '{s}'. Valid values are: full-reindex, continuous-monitoring, read-only"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing_mode_properties() {
        assert!(IndexingMode::FullReindex.allows_indexing());
        assert!(!IndexingMode::FullReindex.allows_file_watching());
        assert!(!IndexingMode::FullReindex.is_read_only());

        assert!(IndexingMode::ContinuousMonitoring.allows_indexing());
        assert!(IndexingMode::ContinuousMonitoring.allows_file_watching());
        assert!(!IndexingMode::ContinuousMonitoring.is_read_only());

        assert!(!IndexingMode::ReadOnly.allows_indexing());
        assert!(!IndexingMode::ReadOnly.allows_file_watching());
        assert!(IndexingMode::ReadOnly.is_read_only());
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            "full-reindex".parse::<IndexingMode>().unwrap(),
            IndexingMode::FullReindex
        );
        assert_eq!(
            "continuous-monitoring".parse::<IndexingMode>().unwrap(),
            IndexingMode::ContinuousMonitoring
        );
        assert_eq!(
            "read-only".parse::<IndexingMode>().unwrap(),
            IndexingMode::ReadOnly
        );

        // Test case insensitive
        assert_eq!(
            "FULL".parse::<IndexingMode>().unwrap(),
            IndexingMode::FullReindex
        );
        assert_eq!(
            "Monitor".parse::<IndexingMode>().unwrap(),
            IndexingMode::ContinuousMonitoring
        );

        // Test invalid
        assert!("invalid".parse::<IndexingMode>().is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(IndexingMode::FullReindex.to_string(), "full-reindex");
        assert_eq!(
            IndexingMode::ContinuousMonitoring.to_string(),
            "continuous-monitoring"
        );
        assert_eq!(IndexingMode::ReadOnly.to_string(), "read-only");
    }
}
