use serde::{Deserialize, Serialize};

/// Database information and statistics
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseFile {
    /// File path
    pub path: String,
    /// File size in bytes
    pub size_bytes: Option<u64>,
    /// File type (main database, WAL, journal, etc.)
    pub file_type: String,
}

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

/// Version information for dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyVersions {
    /// janet-ai-retriever version
    pub retriever_version: String,
    /// janet-ai-embed version
    pub embed_version: String,
    /// janet-ai-context version
    pub context_version: String,
    /// Rust version used for compilation
    pub rust_version: String,
    /// Core dependency versions
    pub dependencies: std::collections::HashMap<String, String>,
}
