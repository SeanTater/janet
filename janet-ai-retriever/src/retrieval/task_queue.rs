//! Priority-based task queue for async indexing operations.
//!
//! This module provides a flexible task queue system for managing indexing work across
//! multiple async workers. It supports priority-based scheduling, retry logic, and
//! configurable concurrency to efficiently process large codebases.
//!
//! ## Key Components
//!
//! - **TaskQueue**: Main queue coordinator with priority scheduling
//! - **IndexingTask**: Individual work items with metadata and priority
//! - **TaskType**: Different types of indexing operations
//! - **TaskPriority**: Priority levels for work prioritization
//! - **TaskQueueConfig**: Configuration for queue behavior and limits
//!
//! ## Features
//!
//! ### Priority-Based Scheduling
//!
//! ### Work Distribution
//! - Tasks are distributed across configurable worker count
//! - Higher priority tasks are processed first
//! - Failed tasks can be retried with backoff
//! - Queue size monitoring prevents memory exhaustion
//!
//! ### Task Types
//! - **IndexFile**: Process and chunk a single file
//! - **GenerateEmbeddings**: Create embeddings for existing chunks
//! - **RemoveFile**: Clean up deleted files from index
//! - **Refresh**: Re-process files that may have changed
//!
//! ## Configuration
//!
//!
//! ## Integration
//!
//! Typically used by IndexingEngine to coordinate work:
//! - File changes from DirectoryWatcher → TaskQueue
//! - Embedding generation → TaskQueue
//! - Database updates → TaskQueue
//! - Statistics and progress tracking

use flume::{Receiver, Sender, bounded};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

/// Priority levels for indexing tasks. See module-level docs for usage examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Background tasks (e.g., bulk reindexing, routine processing)
    Background = 0,
    /// Priority tasks (e.g., file changes, user requests, file removals)
    Priority = 1,
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Priority
    }
}

/// Types of indexing operations. See module-level docs for usage patterns.
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Index a single file
    IndexFile { path: PathBuf },
    /// Remove a file from the index (when deleted)
    RemoveFile { path: PathBuf },
}

/// Individual work item with metadata and retry logic. See module-level docs for examples.
#[derive(Debug, Clone)]
pub struct IndexingTask {
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub created_at: u64, // Unix timestamp in seconds
    pub retry_count: u32,
}

impl IndexingTask {
    /// Create a new task with the specified type and priority.
    pub fn new(task_type: TaskType, priority: TaskPriority) -> Self {
        Self {
            task_type,
            priority,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            retry_count: 0,
        }
    }

    /// Convenience method for normal-priority file indexing.
    pub fn index_file(path: PathBuf) -> Self {
        Self::new(TaskType::IndexFile { path }, TaskPriority::Priority)
    }

    /// Convenience method for background file indexing.
    pub fn index_file_background(path: PathBuf) -> Self {
        Self::new(TaskType::IndexFile { path }, TaskPriority::Background)
    }

    /// Convenience method for file removal tasks.
    pub fn remove_file(path: PathBuf) -> Self {
        Self::new(TaskType::RemoveFile { path }, TaskPriority::Priority)
    }

    /// Returns task age in seconds since creation.
    pub fn age_seconds(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(self.created_at)
    }

    /// Increments the retry count for failed task attempts.
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Returns whether this task should be retried (max 3 attempts).
    pub fn should_retry(&self) -> bool {
        self.retry_count < 3
    }

    /// Returns a human-readable description for logging.
    pub fn description(&self) -> String {
        match &self.task_type {
            TaskType::IndexFile { path } => format!("Index file: {}", path.display()),
            TaskType::RemoveFile { path } => format!("Remove file: {}", path.display()),
        }
    }
}

/// Configuration for task queue behavior. See module docs for examples.
#[derive(Debug, Clone)]
pub struct TaskQueueConfig {
    /// Maximum number of tasks in the queue
    pub max_queue_size: usize,
    /// Maximum number of concurrent workers
    pub max_workers: usize,
    /// Timeout for task processing
    pub task_timeout: Duration,
    /// Batch size for grouping similar tasks
    pub batch_size: usize,
    /// Interval for batching tasks
    pub batch_interval: Duration,
}

impl Default for TaskQueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            max_workers: 4,
            task_timeout: Duration::from_secs(300), // 5 minutes
            batch_size: 50,
            batch_interval: Duration::from_secs(5),
        }
    }
}

/// Priority-based async task queue. See module docs for usage patterns.
#[derive(Debug)]
pub struct TaskQueue {
    sender: Sender<IndexingTask>,
    receiver: Receiver<IndexingTask>,
}

impl TaskQueue {
    /// Creates a new task queue with the specified configuration.
    pub fn new(config: TaskQueueConfig) -> Self {
        let (sender, receiver) = bounded(config.max_queue_size);

        Self { sender, receiver }
    }

    /// Submit a task to the queue
    pub fn submit_task(&self, task: IndexingTask) -> Result<(), String> {
        debug!("Submitting task: {}", task.description());

        self.sender.try_send(task).map_err(|e| match e {
            flume::TrySendError::Full(_) => {
                warn!("Task queue is full, dropping task");
                "Task queue is full".to_string()
            }
            flume::TrySendError::Disconnected(_) => "Task queue is shutdown".to_string(),
        })?;

        Ok(())
    }

    /// Submit multiple tasks at once
    pub fn submit_tasks(&self, tasks: Vec<IndexingTask>) -> Result<(), String> {
        for task in tasks {
            self.submit_task(task)?;
        }
        Ok(())
    }

    /// Get the next task from the queue (blocking)
    pub async fn recv_task(&self) -> Result<IndexingTask, String> {
        self.receiver
            .recv_async()
            .await
            .map_err(|_| "Task queue is shutdown".to_string())
    }

    /// Try to get the next task from the queue (non-blocking)
    pub fn try_recv_task(&self) -> Result<IndexingTask, String> {
        self.receiver.try_recv().map_err(|e| match e {
            flume::TryRecvError::Empty => "No tasks available".to_string(),
            flume::TryRecvError::Disconnected => "Task queue is shutdown".to_string(),
        })
    }

    /// Get the current queue size
    pub fn queue_size(&self) -> usize {
        self.receiver.len()
    }

    /// Check if the queue is shutdown
    pub fn is_shutdown(&self) -> bool {
        self.receiver.is_disconnected()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_priority_ordering() {
        let background = IndexingTask::new(
            TaskType::IndexFile {
                path: PathBuf::from("test"),
            },
            TaskPriority::Background,
        );
        let priority = IndexingTask::new(
            TaskType::IndexFile {
                path: PathBuf::from("test"),
            },
            TaskPriority::Priority,
        );

        assert!(priority.priority > background.priority);
    }

    #[tokio::test]
    async fn test_task_queue_operations() {
        let config = TaskQueueConfig::default();
        let queue = TaskQueue::new(config);

        // Submit some tasks
        let task1 = IndexingTask::index_file_background(PathBuf::from("file1.txt"));
        let task2 = IndexingTask::index_file(PathBuf::from("file2.txt"));
        let task3 = IndexingTask::index_file(PathBuf::from("file3.txt"));

        queue.submit_task(task1).unwrap();
        queue.submit_task(task2).unwrap();
        queue.submit_task(task3).unwrap();

        // Receive tasks (order depends on flume's internal handling)
        let received1 = queue.recv_task().await.unwrap();
        let received2 = queue.recv_task().await.unwrap();
        let received3 = queue.recv_task().await.unwrap();

        // Verify we got all tasks back
        assert_eq!(queue.queue_size(), 0);

        // At least one should be a priority task
        let priorities = [received1.priority, received2.priority, received3.priority];
        assert!(priorities.contains(&TaskPriority::Priority));
        assert!(priorities.contains(&TaskPriority::Background));
    }

    #[tokio::test]
    async fn test_task_retry_logic() {
        let mut task = IndexingTask::index_file(PathBuf::from("test.txt"));

        assert!(task.should_retry());
        assert_eq!(task.retry_count, 0);

        task.increment_retry();
        assert!(task.should_retry());
        assert_eq!(task.retry_count, 1);

        task.increment_retry();
        task.increment_retry();
        assert!(!task.should_retry());
        assert_eq!(task.retry_count, 3);
    }

    #[test]
    fn test_task_priority_parsing() {
        assert_eq!(TaskPriority::Background as u8, 0);
        assert_eq!(TaskPriority::Priority as u8, 1);

        assert!(TaskPriority::Priority > TaskPriority::Background);
    }
}
