use serde::{Deserialize, Serialize};

/// File system monitoring status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemStatus {
    /// Is file watching currently active?
    pub file_watching_active: bool,
    /// Number of directories being monitored
    pub directories_monitored: usize,
    /// File watcher error count
    pub watcher_error_count: usize,
    /// Last file change event timestamp
    pub last_change_event_timestamp: Option<i64>,
    /// Supported file system types
    pub supported_file_systems: Vec<String>,
    /// File system type of base directory
    pub base_directory_fs_type: Option<String>,
    /// Recent file change events (last 10)
    pub recent_events: Vec<FileChangeEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChangeEvent {
    /// Event type (Created, Modified, Deleted, etc.)
    pub event_type: String,
    /// File path that changed
    pub file_path: String,
    /// Timestamp when event occurred
    pub timestamp: i64,
    /// Whether the event was processed successfully
    pub processed_successfully: bool,
}

/// Stale files information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaleFilesInfo {
    /// Files modified after last index update
    pub modified_files: Vec<StaleFileEntry>,
    /// Files added but not yet indexed
    pub unindexed_files: Vec<StaleFileEntry>,
    /// Files deleted but still in index (estimated)
    pub deleted_files_in_index: Vec<StaleFileEntry>,
    /// Recommended reindex candidates
    pub reindex_candidates: Vec<String>,
    /// Total estimated stale files count
    pub total_stale_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaleFileEntry {
    /// File path
    pub file_path: String,
    /// Last modified timestamp
    pub last_modified_timestamp: Option<i64>,
    /// Reason for being considered stale
    pub staleness_reason: String,
    /// Priority for reindexing (1-10, higher is more urgent)
    pub reindex_priority: u8,
}
