use serde::{Deserialize, Serialize};

/// Basic file system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemStatus {
    /// Base directory exists and is accessible
    pub base_directory_accessible: bool,
}

// Removed FileChangeEvent - not tracked

/// Basic stale files information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaleFilesInfo {
    /// Number of pending tasks in indexing queue
    pub pending_tasks: usize,
}

// Removed StaleFileEntry - complex tracking not implemented
