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
//! ```rust,no_run
//! use janet_ai_retriever::retrieval::task_queue::{TaskQueue, IndexingTask, TaskPriority, TaskType, TaskQueueConfig};
//! use std::path::Path;
//! use std::time::Duration;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = TaskQueueConfig {
//!     max_queue_size: 1000,
//!     max_workers: 4,
//!     task_timeout: Duration::from_secs(30),
//!     batch_size: 10,
//!     batch_interval: Duration::from_millis(100),
//! };
//! let mut queue = TaskQueue::new(config);
//!
//! // High priority: recently modified files
//! let high_priority_task = IndexingTask::new(
//!     TaskType::IndexFile { path: Path::new("src/main.rs").to_path_buf() },
//!     TaskPriority::High
//! );
//!
//! // Background: routine indexing
//! let background_task = IndexingTask::new(
//!     TaskType::IndexFile { path: Path::new("docs/readme.md").to_path_buf() },
//!     TaskPriority::Background
//! );
//! # Ok(())
//! # }
//! ```
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
//! ```rust,no_run
//! use janet_ai_retriever::retrieval::task_queue::{TaskQueue, TaskQueueConfig};
//! use std::time::Duration;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = TaskQueueConfig {
//!     max_workers: 8,                    // More workers for faster processing
//!     max_queue_size: 10000,             // Prevent memory overflow
//!     task_timeout: Duration::from_secs(30),
//!     batch_size: 16,                    // Batch size for efficiency
//!     batch_interval: Duration::from_millis(100),
//! };
//!
//! let queue = TaskQueue::new(config);
//! # Ok(())
//! # }
//! ```
//!
//! ## Integration
//!
//! Typically used by IndexingEngine to coordinate work:
//! - File changes from DirectoryWatcher → TaskQueue
//! - Embedding generation → TaskQueue
//! - Database updates → TaskQueue
//! - Statistics and progress tracking

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, Notify, mpsc};
use tracing::{debug, warn};

/// Priority levels for indexing tasks
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

/// Types of indexing tasks
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Index a single file
    IndexFile { path: PathBuf },
    /// Remove a file from the index (when deleted)
    RemoveFile { path: PathBuf },
}

/// A task in the indexing queue
#[derive(Debug, Clone)]
pub struct IndexingTask {
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub created_at: u64, // Unix timestamp in seconds
    pub retry_count: u32,
}

impl IndexingTask {
    /// Create a new indexing task
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

    /// Create a priority task for indexing a single file
    pub fn index_file(path: PathBuf) -> Self {
        Self::new(TaskType::IndexFile { path }, TaskPriority::Priority)
    }

    /// Create a background task for indexing a single file
    pub fn index_file_background(path: PathBuf) -> Self {
        Self::new(TaskType::IndexFile { path }, TaskPriority::Background)
    }

    /// Create a task for removing a file from the index
    pub fn remove_file(path: PathBuf) -> Self {
        Self::new(TaskType::RemoveFile { path }, TaskPriority::Priority)
    }

    /// Get the age of this task in seconds
    pub fn age_seconds(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(self.created_at)
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Check if task should be retried (max 3 retries)
    pub fn should_retry(&self) -> bool {
        self.retry_count < 3
    }

    /// Get a description of the task for logging
    pub fn description(&self) -> String {
        match &self.task_type {
            TaskType::IndexFile { path } => format!("Index file: {}", path.display()),
            TaskType::RemoveFile { path } => format!("Remove file: {}", path.display()),
        }
    }
}

/// Wrapper for priority queue ordering
#[derive(Debug)]
struct PriorityTask {
    task: IndexingTask,
    /// Secondary priority based on file modification time for files
    /// More recently modified files get higher priority
    file_priority: u64,
}

impl PriorityTask {
    fn new(task: IndexingTask) -> Self {
        let file_priority = match &task.task_type {
            TaskType::IndexFile { path } | TaskType::RemoveFile { path } => {
                // Use file modification time as secondary priority
                std::fs::metadata(path)
                    .and_then(|m| m.modified())
                    .map(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
                    .unwrap_or(0)
            }
        };

        Self {
            task,
            file_priority,
        }
    }
}

impl PartialEq for PriorityTask {
    fn eq(&self, other: &Self) -> bool {
        self.task.priority == other.task.priority && self.file_priority == other.file_priority
    }
}

impl Eq for PriorityTask {}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: task priority (higher is better)
        match self.task.priority.cmp(&other.task.priority) {
            Ordering::Equal => {
                // Secondary: file modification time (more recent is better)
                self.file_priority.cmp(&other.file_priority)
            }
            other => other,
        }
    }
}

/// Configuration for the task queue
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

/// A priority-based task queue for managing indexing operations
#[derive(Debug)]
pub struct TaskQueue {
    config: TaskQueueConfig,
    queue: Arc<Mutex<BinaryHeap<PriorityTask>>>,
    task_sender: mpsc::UnboundedSender<IndexingTask>,
    task_receiver: Arc<Mutex<mpsc::UnboundedReceiver<IndexingTask>>>,
    shutdown_notify: Arc<Notify>,
    is_shutdown: Arc<Mutex<bool>>,
}

impl TaskQueue {
    /// Create a new task queue with the given configuration
    pub fn new(config: TaskQueueConfig) -> Self {
        let (task_sender, task_receiver) = mpsc::unbounded_channel();

        Self {
            config,
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            task_sender,
            task_receiver: Arc::new(Mutex::new(task_receiver)),
            shutdown_notify: Arc::new(Notify::new()),
            is_shutdown: Arc::new(Mutex::new(false)),
        }
    }

    /// Submit a task to the queue
    pub async fn submit_task(&self, task: IndexingTask) -> Result<(), String> {
        if *self.is_shutdown.lock().await {
            return Err("Task queue is shutdown".to_string());
        }

        // Check queue size limit
        {
            let queue = self.queue.lock().await;
            if queue.len() >= self.config.max_queue_size {
                warn!("Task queue is full, dropping task: {}", task.description());
                return Err("Task queue is full".to_string());
            }
        }

        debug!("Submitting task: {}", task.description());

        self.task_sender
            .send(task)
            .map_err(|e| format!("Failed to submit task: {e}"))?;

        Ok(())
    }

    /// Submit multiple tasks at once
    pub async fn submit_tasks(&self, tasks: Vec<IndexingTask>) -> Result<(), String> {
        for task in tasks {
            self.submit_task(task).await?;
        }
        Ok(())
    }

    /// Get the next highest priority task from the queue
    pub async fn pop_task(&self) -> Option<IndexingTask> {
        let mut queue = self.queue.lock().await;
        queue.pop().map(|pt| pt.task)
    }

    /// Get the current queue size
    pub async fn queue_size(&self) -> usize {
        let queue = self.queue.lock().await;
        queue.len()
    }

    /// Start the queue processor that moves tasks from the channel to the priority queue
    pub async fn start_processor(&self) {
        let queue = Arc::clone(&self.queue);
        let receiver = Arc::clone(&self.task_receiver);
        let shutdown_notify = Arc::clone(&self.shutdown_notify);
        let is_shutdown = Arc::clone(&self.is_shutdown);

        tokio::spawn(async move {
            let mut receiver = receiver.lock().await;

            loop {
                tokio::select! {
                    task = receiver.recv() => {
                        match task {
                            Some(task) => {
                                let mut queue = queue.lock().await;
                                queue.push(PriorityTask::new(task));
                            }
                            None => {
                                debug!("Task channel closed, stopping processor");
                                break;
                            }
                        }
                    }
                    _ = shutdown_notify.notified() => {
                        debug!("Shutdown signal received, stopping processor");
                        break;
                    }
                }
            }

            let mut shutdown = is_shutdown.lock().await;
            *shutdown = true;
        });
    }

    /// Shutdown the task queue
    pub async fn shutdown(&self) {
        debug!("Shutting down task queue");
        self.shutdown_notify.notify_waiters();

        // Wait for shutdown to complete
        while !*self.is_shutdown.lock().await {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        debug!("Task queue shutdown complete");
    }

    /// Check if the queue is shutdown
    pub async fn is_shutdown(&self) -> bool {
        *self.is_shutdown.lock().await
    }

    /// Clear all tasks from the queue
    pub async fn clear(&self) {
        let mut queue = self.queue.lock().await;
        queue.clear();
        debug!("Task queue cleared");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_priority_ordering() {
        let background = PriorityTask::new(IndexingTask::new(
            TaskType::IndexFile {
                path: PathBuf::from("test"),
            },
            TaskPriority::Background,
        ));
        let priority = PriorityTask::new(IndexingTask::new(
            TaskType::IndexFile {
                path: PathBuf::from("test"),
            },
            TaskPriority::Priority,
        ));

        assert!(priority > background);
    }

    #[tokio::test]
    async fn test_task_queue_operations() {
        let config = TaskQueueConfig::default();
        let queue = TaskQueue::new(config);

        // Start the processor
        queue.start_processor().await;

        // Submit some tasks
        let task1 = IndexingTask::index_file_background(PathBuf::from("file1.txt"));
        let task2 = IndexingTask::index_file(PathBuf::from("file2.txt"));
        let task3 = IndexingTask::index_file(PathBuf::from("file3.txt"));

        queue.submit_task(task1).await.unwrap();
        queue.submit_task(task2).await.unwrap();
        queue.submit_task(task3).await.unwrap();

        // Give processor time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Priority tasks should come first (in any order since they have same priority)
        let next_task = queue.pop_task().await.unwrap();
        assert_eq!(next_task.priority, TaskPriority::Priority);

        let next_task = queue.pop_task().await.unwrap();
        assert_eq!(next_task.priority, TaskPriority::Priority);

        // Background priority comes last
        let next_task = queue.pop_task().await.unwrap();
        assert_eq!(next_task.priority, TaskPriority::Background);

        queue.shutdown().await;
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
