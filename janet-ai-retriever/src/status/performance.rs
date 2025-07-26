use serde::{Deserialize, Serialize};

/// Basic search functionality status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPerformanceStats {
    /// Whether search functionality is available
    pub search_available: bool,
}

// Removed SearchQualityMetrics and SearchErrorRates - not tracked

/// Basic indexing functionality status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingPerformanceStats {
    /// Whether indexing is operational
    pub indexing_operational: bool,
}

// Removed DiskIOStats - not tracked
