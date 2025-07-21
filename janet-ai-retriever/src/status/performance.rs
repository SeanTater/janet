use serde::{Deserialize, Serialize};

/// Search performance statistics
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQualityMetrics {
    /// Average number of results returned
    pub average_results_count: Option<f64>,
    /// Average relevance score (0-1)
    pub average_relevance_score: Option<f64>,
    /// Percentage of searches returning zero results
    pub zero_results_percentage: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchErrorRates {
    /// Semantic search error rate (0-1)
    pub semantic_search_error_rate: f32,
    /// Text search error rate (0-1)
    pub text_search_error_rate: f32,
    /// Total queries processed
    pub total_queries_processed: usize,
}

/// Indexing performance statistics
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOStats {
    /// Bytes read per second (recent average)
    pub bytes_read_per_second: Option<u64>,
    /// Bytes written per second (recent average)
    pub bytes_written_per_second: Option<u64>,
    /// Total disk space used for indexing in bytes
    pub total_disk_space_used_bytes: Option<u64>,
}
