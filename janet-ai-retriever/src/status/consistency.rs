use serde::{Deserialize, Serialize};

/// Index consistency check results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConsistencyReport {
    /// Overall consistency status
    pub overall_status: ConsistencyStatus,
    /// Checks performed
    pub checks_performed: Vec<ConsistencyCheck>,
    /// Summary of issues found
    pub issues_summary: IssuesSummary,
    /// Timestamp when check was performed
    pub check_timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyStatus {
    Healthy,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyCheck {
    /// Name of the check
    pub check_name: String,
    /// Check status
    pub status: ConsistencyStatus,
    /// Description of what was checked
    pub description: String,
    /// Number of items checked
    pub items_checked: usize,
    /// Number of issues found
    pub issues_found: usize,
    /// Details about issues (if any)
    pub issue_details: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssuesSummary {
    /// Total issues found
    pub total_issues: usize,
    /// Critical issues requiring immediate attention
    pub critical_issues: usize,
    /// Warning issues that should be addressed
    pub warning_issues: usize,
    /// Recommendations for fixing issues
    pub recommendations: Vec<String>,
}
