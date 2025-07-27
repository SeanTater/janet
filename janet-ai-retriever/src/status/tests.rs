#[cfg(test)]
mod test_suite {
    use super::super::api::{ConsistencyStatus, NetworkHealth, StatusApi};
    use super::super::types::HealthStatus;
    use crate::retrieval::indexing_engine::{IndexingEngine, IndexingEngineConfig};
    use anyhow::Result;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_get_index_statistics() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let stats = StatusApi::get_index_statistics(enhanced_index).await?;

        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_embeddings, 0);
        assert_eq!(stats.models_count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_indexing_status() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let engine = IndexingEngine::new_memory(config).await?;

        let status = StatusApi::get_indexing_status(&engine).await?;

        assert!(!status.is_running);
        assert_eq!(status.queue_size, 0);
        assert_eq!(status.error_count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_index_health() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let health = StatusApi::get_index_health(enhanced_index).await?;

        assert!(health.database_connected);
        assert!(health.database_integrity_ok);
        assert!(matches!(health.overall_status, HealthStatus::Healthy));

        Ok(())
    }

    #[tokio::test]
    async fn test_get_database_info() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let db_info = StatusApi::get_database_info(enhanced_index, temp_dir.path()).await?;

        assert_eq!(db_info.database_type, "SQLite");
        assert!(db_info.database_version.is_some());
        assert!(db_info.sqlite_info.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_get_dependency_versions() -> Result<()> {
        let versions = StatusApi::get_dependency_versions().await?;

        assert!(!versions.retriever_version.is_empty());
        assert!(versions.dependencies.is_empty()); // No individual deps tracked

        Ok(())
    }

    #[tokio::test]
    async fn test_validate_index_consistency() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let report = StatusApi::validate_index_consistency(enhanced_index).await?;

        assert!(!report.checks_performed.is_empty());
        // Empty database will show as Warning status
        assert!(matches!(report.overall_status, ConsistencyStatus::Warning));
        assert_eq!(report.issues_summary.total_issues, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_file_system_status() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let fs_status = StatusApi::get_file_system_status(&config).await?;

        assert!(fs_status.base_directory_accessible); // Directory should exist
        // Removed recent_events check - field no longer exists

        Ok(())
    }

    #[tokio::test]
    async fn test_get_search_performance_stats() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let engine = IndexingEngine::new_memory(config).await?;
        let enhanced_index = engine.get_enhanced_index();

        let search_stats = StatusApi::get_search_performance_stats(enhanced_index).await?;

        assert!(search_stats.search_available); // Search functionality available

        Ok(())
    }

    #[tokio::test]
    async fn test_get_indexing_performance_stats() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let engine = IndexingEngine::new_memory(config).await?;

        let indexing_stats = StatusApi::get_indexing_performance_stats(&engine).await?;

        assert!(indexing_stats.indexing_operational); // Indexing functionality available

        Ok(())
    }

    #[tokio::test]
    async fn test_get_stale_files() -> Result<()> {
        let temp_dir = tempdir()?;
        let config =
            IndexingEngineConfig::new("test-repo".to_string(), temp_dir.path().to_path_buf());

        let engine = IndexingEngine::new_memory(config.clone()).await?;

        let stale_files = StatusApi::get_stale_files(&engine, &config).await?;

        // Basic stale files info only tracks pending tasks
        assert_eq!(stale_files.pending_tasks, 0); // No pending tasks in test

        Ok(())
    }

    #[tokio::test]
    async fn test_get_network_status() -> Result<()> {
        let network_status = StatusApi::get_network_status().await?;

        assert!(matches!(
            network_status.overall_network_health,
            NetworkHealth::Healthy
        ));
        // Network status simplified - no connectivity testing
        assert!(matches!(
            network_status.overall_network_health,
            NetworkHealth::Healthy
        ));

        Ok(())
    }
}
