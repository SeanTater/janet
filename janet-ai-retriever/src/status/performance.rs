//! Performance metrics and statistics for janet-ai-retriever
//!
//! This module provides data structures for monitoring and reporting performance metrics
//! including search response times, indexing throughput, resource usage, and quality
//! metrics. Used by the status API to provide detailed performance diagnostics.
//!
//! ## Key Components
//!
//! - **SearchPerformanceStats**: Search response times, quality metrics, and error rates
//! - **SearchQualityMetrics**: Result relevance and user experience metrics
//! - **IndexingPerformanceStats**: File processing throughput and resource usage
//! - **DiskIOStats**: Disk I/O performance and storage utilization
//! - **SearchErrorRates**: Error tracking by operation type
//!
//! ## Usage
//!
//! ```rust
//! use janet_ai_retriever::status::performance::{SearchPerformanceStats, SearchQualityMetrics};
//! use std::collections::HashMap;
//!
//! // Performance stats would typically be collected during system operation
//! let stats = SearchPerformanceStats {
//!     average_response_time_ms: Some(150.5),
//!     result_quality_metrics: SearchQualityMetrics {
//!         average_results_count: Some(8.2),
//!         average_relevance_score: Some(0.75),
//!         zero_results_percentage: Some(5.2),
//!     },
//!     cache_hit_rate_percentage: Some(85.3),
//!     common_query_patterns: vec!["function".to_string(), "class".to_string()],
//!     error_rates: Default::default(),
//! };
//! ```

use serde::{Deserialize, Serialize};

/// Search performance and quality statistics. See module docs for usage examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPerformanceStats {
    /// Average search response time in milliseconds (last 100 queries)
    pub average_response_time_ms: Option<f64>,
    /// Search result quality metrics
    pub result_quality_metrics: SearchQualityMetrics,
    /// Cache hit rate percentage (0-100)
    pub cache_hit_rate_percentage: Option<f32>,
    /// Most common query patterns
    pub common_query_patterns: Vec<String>,
    /// Error rates by operation type
    pub error_rates: SearchErrorRates,
}

/// Search result quality and relevance metrics. See module docs for details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQualityMetrics {
    /// Average number of results returned
    pub average_results_count: Option<f64>,
    /// Average relevance score (0-1)
    pub average_relevance_score: Option<f64>,
    /// Percentage of searches returning zero results
    pub zero_results_percentage: Option<f32>,
}

/// Error rates by search operation type. See module docs for details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchErrorRates {
    /// Semantic search error rate (0-1)
    pub semantic_search_error_rate: f32,
    /// Text search error rate (0-1)
    pub text_search_error_rate: f32,
    /// Total queries processed
    pub total_queries_processed: usize,
}

/// File indexing throughput and resource usage. See module docs for usage examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingPerformanceStats {
    /// Files processed per minute (recent average)
    pub files_per_minute: Option<f64>,
    /// Average processing time per file type in milliseconds
    pub processing_time_by_file_type: std::collections::HashMap<String, f64>,
    /// Embedding generation speed (embeddings per second)
    pub embeddings_per_second: Option<f64>,
    /// Disk I/O statistics
    pub disk_io_stats: DiskIOStats,
    /// Memory usage during indexing in bytes
    pub peak_memory_usage_bytes: Option<u64>,
}

/// Disk I/O performance and storage metrics. See module docs for details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOStats {
    /// Bytes read per second (recent average)
    pub bytes_read_per_second: Option<u64>,
    /// Bytes written per second (recent average)
    pub bytes_written_per_second: Option<u64>,
    /// Total disk space used for indexing in bytes
    pub total_disk_space_used_bytes: Option<u64>,
}

impl Default for SearchErrorRates {
    fn default() -> Self {
        Self {
            semantic_search_error_rate: 0.0,
            text_search_error_rate: 0.0,
            total_queries_processed: 0,
        }
    }
}
