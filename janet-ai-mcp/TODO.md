# Janet AI MCP Server - TODO

This document outlines enhancements needed in `janet-ai-retriever` and the MCP server to provide comprehensive status reporting and improved debugging capabilities.

## janet-ai-retriever API Enhancements Needed

### Index Statistics & Health
- [ ] `get_index_statistics()` - Return comprehensive index stats:
  - Total files indexed
  - Total chunks created
  - Total embeddings generated
  - Index database size (bytes)
  - Last indexing operation timestamp
  - Index schema version
  - Corruption check status

- [ ] `get_indexing_status()` - Return current indexing operation status:
  - Is indexing currently running?
  - Queue size (pending files to process)
  - Current file being processed
  - Progress percentage
  - Estimated time remaining
  - Error count during current/last run

- [ ] `get_index_health()` - Return health check information:
  - Database connectivity status
  - File permissions for index directory
  - Disk space available for index growth
  - Memory usage estimates
  - Database integrity check results

### Configuration & Model Information
- [ ] `get_indexing_config()` - Return current indexing configuration:
  - Chunk size and overlap settings
  - File type patterns included/excluded
  - Maximum file size limits
  - Indexing mode (full/incremental/read-only)
  - Worker thread count

- [ ] `get_embedding_model_info()` - Return embedding model details:
  - Model name and version
  - Model dimensions
  - Model download status
  - Model file size and location
  - Supported languages/domains
  - Normalization settings

- [ ] `get_supported_file_types()` - Return list of file types that can be indexed

### Performance Metrics
- [ ] `get_search_performance_stats()` - Return search performance metrics:
  - Average search response time (last 100 queries)
  - Search result quality metrics
  - Cache hit rates
  - Most common query patterns
  - Error rates by operation type

- [ ] `get_indexing_performance_stats()` - Return indexing performance metrics:
  - Files processed per minute
  - Average processing time per file type
  - Embedding generation speed
  - Disk I/O statistics
  - Memory usage during indexing

### File System Integration
- [ ] `get_file_system_status()` - Return file system monitoring status:
  - Is file watching active?
  - Number of directories being monitored
  - Recent file change events
  - File watcher error count
  - Supported file system types

- [ ] `get_stale_files()` - Return list of files that have changed since last index:
  - Files modified after last index update
  - Files added but not yet indexed
  - Files deleted but still in index
  - Recommended reindex candidates

### Database & Storage
- [ ] `get_database_info()` - Return database-specific information:
  - Database type and version
  - Connection pool status
  - Query performance statistics
  - Database file locations and sizes
  - Backup status and last backup time

- [ ] `validate_index_consistency()` - Run consistency checks:
  - Verify all chunks have corresponding files
  - Check for orphaned embeddings
  - Validate file hash consistency
  - Report missing or corrupted data

### Network & Dependencies
- [ ] `get_network_status()` - Return network-dependent feature status:
  - Model download connectivity
  - Hugging Face Hub access
  - Proxy configuration status
  - SSL certificate validation

- [ ] `get_dependency_versions()` - Return version information:
  - janet-ai-retriever version
  - janet-ai-embed version
  - janet-ai-context version
  - Core dependency versions (fastembed, sqlx, etc.)

## MCP Server Status Endpoint Enhancements

### Current System Status
- [ ] Display current working directory and permissions
- [ ] Show available disk space and memory
- [ ] Report janet-ai-mcp server version and uptime
- [ ] Display Rust version and compilation target

### Index Infrastructure Status
- [ ] Index database existence and accessibility
- [ ] Index freshness (last update time vs file system changes)
- [ ] Index completeness (percentage of files indexed)
- [ ] Background indexing operation status

### Search Functionality Status
- [ ] Regex search: Always available ✓
- [ ] Semantic search: Status based on index and embeddings availability
- [ ] File content search: Status based on index availability
- [ ] Performance benchmarks for each search type

### Configuration Summary
- [ ] Root directory and scan depth
- [ ] File type filters and exclusions
- [ ] Indexing schedule and automation status
- [ ] Embedding model configuration and status

### Troubleshooting Information
- [ ] Common error scenarios and solutions
- [ ] Required setup steps if infrastructure missing
- [ ] Performance tuning recommendations
- [ ] Log file locations and recent error summary

### Operational Metrics
- [ ] Recent search query statistics
- [ ] System resource usage trends
- [ ] Error rates and recovery status
- [ ] Cache utilization and effectiveness

## Implementation Priority

### High Priority (MVP)
1. Basic index statistics (file/chunk/embedding counts)
2. Index database existence and health checks
3. Embedding model status and configuration
4. Current indexing operation status

### Medium Priority (Enhanced Debugging)
1. Performance metrics and benchmarks
2. File system monitoring status
3. Configuration validation and reporting
4. Stale file detection

### Low Priority (Advanced Features)
1. Historical performance trends
2. Predictive indexing recommendations
3. Advanced troubleshooting automation
4. Integration testing utilities

## API Design Considerations

- All status functions should be fast (< 100ms) and cacheable
- Functions should gracefully handle missing or corrupted index data
- Return structured data (JSON/structs) rather than formatted strings when possible
- Include timestamp metadata for cache invalidation
- Provide both summary and detailed views for complex information
- Support async operations for potentially slow operations (consistency checks)
