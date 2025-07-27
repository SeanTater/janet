//! Database diagnostics and version information for janet-ai-retriever
//!
//! This module provides data structures for reporting database health, connection status,
//! file information, and dependency versions. Used by the status API to provide detailed
//! diagnostics about the SQLite database and system dependencies.
//!
//! ## Key Components
//!
//! - **DatabaseInfo**: Complete database status and configuration
//! - **ConnectionPoolStatus**: SQLite connection pool monitoring
//! - **DatabaseFile**: Individual database file information (main, WAL, journal)
//! - **SqliteInfo**: SQLite-specific configuration and statistics
//! - **DependencyVersions**: Version information for all system dependencies
//!
//! ## Usage
//!

use serde::{Deserialize, Serialize};

/// Complete database status and configuration. See module docs for usage examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseInfo {
    /// Database type and version
    pub database_type: String,
    /// Database version
    pub database_version: Option<String>,
    /// Connection pool status
    pub connection_pool_status: ConnectionPoolStatus,
    /// Database file locations and sizes
    pub database_files: Vec<DatabaseFile>,
    /// Total database size in bytes
    pub total_size_bytes: Option<u64>,
    /// SQLite specific information
    pub sqlite_info: Option<SqliteInfo>,
}

/// SQLite connection pool status. See module docs for details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolStatus {
    /// Total connections in pool
    pub total_connections: u32,
    /// Active connections
    pub active_connections: u32,
    /// Maximum connections allowed
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connection_timeout_seconds: Option<u64>,
}

/// Individual database file information. See module docs for details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseFile {
    /// File path
    pub path: String,
    /// File size in bytes
    pub size_bytes: Option<u64>,
    /// File type (main database, WAL, journal, etc.)
    pub file_type: String,
}

/// SQLite-specific configuration and statistics. See module docs for details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqliteInfo {
    /// SQLite version
    pub version: String,
    /// Journal mode (WAL, DELETE, etc.)
    pub journal_mode: Option<String>,
    /// Synchronous mode
    pub synchronous_mode: Option<String>,
    /// Page size
    pub page_size: Option<u64>,
    /// Total pages
    pub page_count: Option<u64>,
}

/// System and dependency version information. See module docs for usage examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyVersions {
    /// janet-ai-retriever version
    pub retriever_version: String,
    /// Rust version used for compilation
    pub rust_version: String,
    /// Core dependency versions
    pub dependencies: std::collections::HashMap<String, String>,
}
