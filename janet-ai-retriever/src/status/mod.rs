//! System diagnostics and health monitoring for janet-ai-retriever
//!
//! This module provides comprehensive system status reporting capabilities, including
//! database health, filesystem monitoring, performance metrics, and consistency checking.
//! It serves as a unified diagnostic interface for monitoring and troubleshooting the
//! janet-ai-retriever system.
//!
//! ## Key Modules
//!
//! - **[`api`]**: StatusApi trait and implementations for collecting diagnostic data
//! - **[`consistency`]**: Index consistency checking and validation
//! - **[`database`]**: Database health, connection pools, and version information
//! - **[`filesystem`]**: File monitoring, change events, and stale file detection
//! - **[`network`]**: Network status and connectivity diagnostics
//! - **[`performance`]**: Performance metrics, timing, and resource usage
//! - **[`types`]**: Common status types and report structures
//!
//! ## Usage
//!
//! ```rust,no_run
//! use janet_ai_retriever::status::{StatusReport, StatusApi};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create status API implementation
//! // let status_api = /* ... */;
//!
//! // Generate comprehensive status report
//! // let report = status_api.generate_status_report().await?;
//! // println!("System status: {:?}", report.overall_status);
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! StatusApi ──┬── Database Diagnostics
//!             ├── Filesystem Monitoring
//!             ├── Performance Metrics
//!             ├── Consistency Checking
//!             └── Network Status
//!                     ↓
//!             Unified StatusReport
//! ```

// Status API modules
pub mod api;
pub mod database;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for easy access
pub use api::*;
pub use database::*;
pub use types::*;
