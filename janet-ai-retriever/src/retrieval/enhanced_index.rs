//! Enhanced file index with metadata tracking and embedding model compatibility.
//!
//! This module provides a high-level wrapper around [`FileIndex`] that adds metadata
//! tracking, embedding model management, and version compatibility checks. It's designed
//! for production use where you need to track multiple embedding models, system versions,
//! and ensure compatibility across upgrades.
//!
//! ## Key Components
//!
//! - **EnhancedFileIndex**: Wrapper around FileIndex with metadata capabilities
//! - **EmbeddingModelMetadata**: Tracks embedding model configurations and compatibility
//! - **IndexMetadata**: Tracks system versions and repository information
//! - **IndexStats**: Provides statistics about the indexed data
//!
//! ## Additional Database Schema
//!
//! Beyond the base FileIndex schema, this adds:
//!
//! ```sql
//! -- Index metadata: tracks system versions and repository info
//! CREATE TABLE index_metadata (
//!     id INTEGER PRIMARY KEY,
//!     repository TEXT UNIQUE,          -- repository identifier
//!     retriever_version TEXT,          -- janet-ai-retriever version
//!     context_version TEXT,            -- janet-ai-context version
//!     embed_version TEXT,              -- janet-ai-embed version
//!     created_at INTEGER,              -- unix timestamp
//!     updated_at INTEGER,              -- unix timestamp
//!     metadata_json TEXT               -- additional metadata
//! );
//!
//! -- Embedding models: tracks different embedding configurations
//! CREATE TABLE embedding_models (
//!     model_id TEXT PRIMARY KEY,       -- unique model identifier
//!     model_name TEXT,                 -- human-readable name
//!     provider TEXT,                   -- embedding provider
//!     dimension INTEGER,               -- vector dimension
//!     model_version TEXT,              -- model version
//!     normalized BOOLEAN,              -- whether vectors are normalized
//!     config_json TEXT                 -- additional configuration
//! );
//!
//! -- Enhanced chunks table adds model tracking
//! ALTER TABLE chunks ADD COLUMN model_id TEXT REFERENCES embedding_models(model_id);
//! ```
//!
//! ## Features
//!
//! ### Model Compatibility Checking
//! ```rust,no_run
//! use janet_ai_retriever::retrieval::enhanced_index::{EnhancedFileIndex, EmbeddingModelMetadata};
//! use std::path::Path;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let index = EnhancedFileIndex::open(Path::new(".")).await?;
//!
//! let model = EmbeddingModelMetadata::new(
//!     "snowflake-arctic-embed-xs".to_string(),
//!     "fastembed".to_string(),
//!     384
//! ).with_normalized(true);
//!
//! // Check if model is compatible with existing embeddings
//! let compatible = index.check_model_compatibility(&model).await?;
//! if !compatible {
//!     // Need to re-embed with new model
//!     index.delete_chunks_by_model(&model.model_id()).await?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Multi-Model Support
//! ```rust,no_run
//! # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
//! # use std::path::Path;
//! # async fn example() -> anyhow::Result<()> {
//! let index = EnhancedFileIndex::open(Path::new(".")).await?;
//!
//! // Get chunks from a specific model
//! let model_chunks = index.get_chunks_by_model("fastembed:snowflake-arctic-embed-xs:latest:384:norm").await?;
//!
//! // Search with similarity
//! let query_embedding = vec![half::f16::from_f32(0.1); 384];
//! let similar_chunks = index.search_similar_chunks(&query_embedding, 10).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## When to Use
//!
//! - **Production systems** that need model versioning
//! - **Multi-model environments** with different embedding strategies
//! - **Long-lived indexes** that need migration support
//! - **Systems requiring audit trails** of embedding model changes
//!
//! For simple use cases, [`FileIndex`] may be sufficient.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::path::Path;

use super::file_index::{ChunkRef, FileIndex};

/// Metadata about the embedding model used for generating embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelMetadata {
    /// Name of the embedding model (e.g., "snowflake-arctic-embed-xs")
    pub model_name: String,
    /// Provider of the embedding model (e.g., "fastembed")
    pub provider: String,
    /// Dimension of the embedding vectors
    pub dimension: usize,
    /// Version/revision of the model
    pub model_version: Option<String>,
    /// Whether embeddings are normalized
    pub normalized: bool,
    /// Additional model-specific configuration
    pub config: HashMap<String, String>,
}

impl EmbeddingModelMetadata {
    pub fn new(model_name: String, provider: String, dimension: usize) -> Self {
        Self {
            model_name,
            provider,
            dimension,
            model_version: None,
            normalized: false,
            config: HashMap::new(),
        }
    }

    pub fn with_version(mut self, version: String) -> Self {
        self.model_version = Some(version);
        self
    }

    pub fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    pub fn with_config(mut self, key: String, value: String) -> Self {
        self.config.insert(key, value);
        self
    }

    /// Create a unique identifier for this model configuration
    pub fn model_id(&self) -> String {
        let version_part = self.model_version.as_deref().unwrap_or("latest");
        let normalized_part = if self.normalized { "norm" } else { "raw" };
        format!(
            "{}:{}:{}:{}:{}",
            self.provider, self.model_name, version_part, self.dimension, normalized_part
        )
    }
}

/// Metadata about the indexing system version and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Version of janet-ai-retriever that created this index
    pub retriever_version: String,
    /// Timestamp when the index was created
    pub created_at: i64,
    /// Timestamp when the index was last updated
    pub updated_at: i64,
    /// Embedding model metadata
    pub embedding_model: Option<EmbeddingModelMetadata>,
    /// Repository name or identifier
    pub repository: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl IndexMetadata {
    pub fn new(repository: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            retriever_version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: now,
            updated_at: now,
            embedding_model: None,
            repository,
            metadata: HashMap::new(),
        }
    }

    pub fn with_embedding_model(mut self, model: EmbeddingModelMetadata) -> Self {
        self.embedding_model = Some(model);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn update_timestamp(&mut self) {
        self.updated_at = chrono::Utc::now().timestamp();
    }
}

/// Enhanced file index with metadata tracking and embedding model compatibility checks
#[derive(Debug)]
pub struct EnhancedFileIndex {
    file_index: FileIndex,
    pool: SqlitePool,
}

impl EnhancedFileIndex {
    /// Create a new enhanced file index with persistent storage.
    ///
    /// This creates an enhanced file index that includes metadata tracking and
    /// embedding model compatibility features. The index uses a `.janet-ai.db`
    /// SQLite database file in the specified directory.
    ///
    /// If the database already exists, it will be opened and any missing tables
    /// will be created automatically.
    ///
    /// # Arguments
    /// * `base` - Directory where the `.janet-ai.db` file will be created
    ///
    /// # Returns
    /// A new EnhancedFileIndex ready for use
    ///
    /// # Errors
    /// - Database connection or creation errors
    /// - File system permission errors
    /// - SQL schema creation errors
    ///
    /// # Example
    /// ```no_run
    /// use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// use std::path::Path;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    /// // Index is ready for metadata operations and file indexing
    /// # Ok(())
    /// # }
    /// ```
    pub async fn open(base: &Path) -> Result<Self> {
        let file_index = FileIndex::open(base).await?;
        let pool = file_index.pool().clone();

        let enhanced = Self {
            file_index,
            pool: pool.clone(),
        };

        // Create additional tables for metadata
        enhanced.create_metadata_tables().await?;

        Ok(enhanced)
    }

    /// Create a new enhanced file index using an in-memory database.
    ///
    /// This creates an enhanced file index that exists only in memory and will be
    /// lost when dropped. This is primarily intended for testing and development
    /// scenarios where persistence is not needed.
    ///
    /// # Arguments
    /// * `base` - Base path used for relative path calculations (not for storage)
    ///
    /// # Returns
    /// A new EnhancedFileIndex using in-memory storage
    ///
    /// # Errors
    /// - Database connection or creation errors
    /// - SQL schema creation errors
    ///
    /// # Example
    /// ```no_run
    /// use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// use std::path::Path;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open_memory(Path::new(".")).await?;
    /// // Index exists in memory only - data will be lost on drop
    /// # Ok(())
    /// # }
    /// ```
    pub async fn open_memory(base: &Path) -> Result<Self> {
        let file_index = FileIndex::open_memory(base).await?;
        let pool = file_index.pool().clone();

        let enhanced = Self {
            file_index,
            pool: pool.clone(),
        };

        // Create additional tables for metadata
        enhanced.create_metadata_tables().await?;

        Ok(enhanced)
    }

    /// Create metadata tables
    async fn create_metadata_tables(&self) -> Result<()> {
        // Create index metadata table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS index_metadata (
                id INTEGER PRIMARY KEY,
                repository TEXT NOT NULL,
                retriever_version TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                metadata_json TEXT,
                CONSTRAINT unique_repo UNIQUE(repository)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create embedding models table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS embedding_models (
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                model_version TEXT,
                normalized BOOLEAN NOT NULL DEFAULT FALSE,
                config_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Add model_id column to chunks table if it doesn't exist
        let add_model_id_result = sqlx::query(
            "ALTER TABLE chunks ADD COLUMN model_id TEXT REFERENCES embedding_models(model_id)",
        )
        .execute(&self.pool)
        .await;

        // Ignore error if column already exists
        if let Err(e) = add_model_id_result {
            if !e.to_string().contains("duplicate column name") {
                return Err(e.into());
            }
        }

        // Create index on model_id
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_chunks_model_id ON chunks(model_id)")
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Get the underlying FileIndex for compatibility
    /// Get a reference to the underlying FileIndex.
    ///
    /// This provides access to the low-level file indexing operations
    /// for cases where you need direct access to the base functionality.
    ///
    /// # Returns
    /// Reference to the underlying FileIndex
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let enhanced_index = EnhancedFileIndex::open(Path::new(".")).await?;
    /// let file_index = enhanced_index.file_index();
    ///
    /// // Use file_index for basic operations
    /// let files = file_index.get_all_files().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn file_index(&self) -> &FileIndex {
        &self.file_index
    }

    /// Initialize or update index metadata for a repository.
    ///
    /// This stores or updates metadata about the index including system versions,
    /// repository information, and any custom metadata. If metadata for the
    /// repository already exists, it will be updated with the new values.
    ///
    /// # Arguments
    /// * `metadata` - Index metadata to store or update
    ///
    /// # Returns
    /// `Ok(())` if the metadata was successfully stored
    ///
    /// # Errors
    /// - JSON serialization errors for metadata
    /// - Database query errors
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::{EnhancedFileIndex, IndexMetadata};
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    ///
    /// let metadata = IndexMetadata::new("my-project".to_string())
    ///     .with_metadata("version".to_string(), "1.0.0".to_string());
    ///
    /// index.upsert_index_metadata(&metadata).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn upsert_index_metadata(&self, metadata: &IndexMetadata) -> Result<()> {
        let metadata_json = serde_json::to_string(&metadata.metadata)?;

        sqlx::query(
            r#"
            INSERT INTO index_metadata
            (repository, retriever_version, created_at, updated_at, metadata_json)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(repository) DO UPDATE SET
                retriever_version = excluded.retriever_version,
                updated_at = excluded.updated_at,
                metadata_json = excluded.metadata_json
            "#,
        )
        .bind(&metadata.repository)
        .bind(&metadata.retriever_version)
        .bind(metadata.created_at)
        .bind(metadata.updated_at)
        .bind(metadata_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get index metadata for a specific repository.
    ///
    /// Retrieves stored metadata about the index including system versions,
    /// timestamps, and any custom metadata that was stored.
    ///
    /// # Arguments
    /// * `repository` - Repository name to look up metadata for
    ///
    /// # Returns
    /// - `Some(IndexMetadata)` if metadata exists for the repository
    /// - `None` if no metadata is found
    ///
    /// # Errors
    /// - Database query errors
    /// - JSON deserialization errors for stored metadata
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    ///
    /// if let Some(metadata) = index.get_index_metadata("my-project").await? {
    ///     println!("Index created at: {}", metadata.created_at);
    ///     println!("Last updated: {}", metadata.updated_at);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_index_metadata(&self, repository: &str) -> Result<Option<IndexMetadata>> {
        let row = sqlx::query("SELECT * FROM index_metadata WHERE repository = ?1")
            .bind(repository)
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            let metadata_json: String = row.get("metadata_json");
            let metadata: HashMap<String, String> =
                serde_json::from_str(&metadata_json).unwrap_or_default();

            Ok(Some(IndexMetadata {
                retriever_version: row.get("retriever_version"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                embedding_model: None, // Will be populated separately if needed
                repository: row.get("repository"),
                metadata,
            }))
        } else {
            Ok(None)
        }
    }

    /// Register an embedding model for use with the index.
    ///
    /// This stores metadata about an embedding model including its provider,
    /// dimensions, normalization settings, and configuration. Models must be
    /// registered before chunks can be associated with them.
    ///
    /// If a model with the same ID already exists, it will be updated.
    ///
    /// # Arguments
    /// * `model` - Embedding model metadata to register
    ///
    /// # Returns
    /// `Ok(())` if the model was successfully registered
    ///
    /// # Errors
    /// - JSON serialization errors for model configuration
    /// - Database query errors
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::{EnhancedFileIndex, EmbeddingModelMetadata};
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    ///
    /// let model = EmbeddingModelMetadata::new(
    ///     "sentence-transformers/all-MiniLM-L6-v2".to_string(),
    ///     "fastembed".to_string(),
    ///     384
    /// ).with_normalized(true);
    ///
    /// index.register_embedding_model(&model).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn register_embedding_model(&self, model: &EmbeddingModelMetadata) -> Result<()> {
        let config_json = serde_json::to_string(&model.config)?;

        sqlx::query(
            r#"
            INSERT INTO embedding_models
            (model_id, model_name, provider, dimension, model_version, normalized, config_json)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(model_id) DO UPDATE SET
                model_name = excluded.model_name,
                provider = excluded.provider,
                dimension = excluded.dimension,
                model_version = excluded.model_version,
                normalized = excluded.normalized,
                config_json = excluded.config_json
            "#,
        )
        .bind(model.model_id())
        .bind(&model.model_name)
        .bind(&model.provider)
        .bind(model.dimension as i64)
        .bind(model.model_version.as_deref())
        .bind(model.normalized)
        .bind(config_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get embedding model metadata by its unique ID.
    ///
    /// Retrieves stored metadata about a specific embedding model including
    /// its configuration, dimensions, and normalization settings.
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier for the embedding model
    ///
    /// # Returns
    /// - `Some(EmbeddingModelMetadata)` if the model exists
    /// - `None` if no model with the given ID is found
    ///
    /// # Errors
    /// - Database query errors
    /// - JSON deserialization errors for model configuration
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    ///
    /// let model_id = "fastembed:sentence-transformers/all-MiniLM-L6-v2:latest:384:norm";
    /// if let Some(model) = index.get_embedding_model(model_id).await? {
    ///     println!("Model: {} ({}D)", model.model_name, model.dimension);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_embedding_model(
        &self,
        model_id: &str,
    ) -> Result<Option<EmbeddingModelMetadata>> {
        let row = sqlx::query("SELECT * FROM embedding_models WHERE model_id = ?1")
            .bind(model_id)
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            let config_json: String = row.get("config_json");
            let config: HashMap<String, String> =
                serde_json::from_str(&config_json).unwrap_or_default();

            Ok(Some(EmbeddingModelMetadata {
                model_name: row.get("model_name"),
                provider: row.get("provider"),
                dimension: row.get::<i64, _>("dimension") as usize,
                model_version: row.get("model_version"),
                normalized: row.get("normalized"),
                config,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get all registered embedding models.
    ///
    /// Retrieves metadata for all embedding models that have been registered
    /// with the index, ordered by creation time (most recent first).
    ///
    /// # Returns
    /// Vector of all registered embedding model metadata
    ///
    /// # Errors
    /// - Database query errors
    /// - JSON deserialization errors for model configurations
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    ///
    /// let models = index.get_all_embedding_models().await?;
    /// for model in models {
    ///     println!("Model: {} ({}D, normalized: {})",
    ///              model.model_name, model.dimension, model.normalized);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_all_embedding_models(&self) -> Result<Vec<EmbeddingModelMetadata>> {
        let rows = sqlx::query("SELECT * FROM embedding_models ORDER BY created_at DESC")
            .fetch_all(&self.pool)
            .await?;

        let mut models = Vec::new();
        for row in rows {
            let config_json: String = row.get("config_json");
            let config: HashMap<String, String> =
                serde_json::from_str(&config_json).unwrap_or_default();

            models.push(EmbeddingModelMetadata {
                model_name: row.get("model_name"),
                provider: row.get("provider"),
                dimension: row.get::<i64, _>("dimension") as usize,
                model_version: row.get("model_version"),
                normalized: row.get("normalized"),
                config,
            });
        }

        Ok(models)
    }

    /// Insert or update chunks with associated embedding model metadata.
    ///
    /// This stores text chunks along with their embeddings and associates them
    /// with a specific embedding model. The model must be registered with
    /// [`register_embedding_model`](Self::register_embedding_model) before calling this method.
    ///
    /// If chunks already exist (based on file_hash, line_start, line_end), they will be updated.
    ///
    /// # Arguments
    /// * `chunks` - Array of chunk references to store
    /// * `model_id` - ID of the embedding model used to generate the embeddings
    ///
    /// # Returns
    /// `Ok(())` if chunks were successfully stored
    ///
    /// # Errors
    /// - Database transaction errors
    /// - Foreign key constraint errors if model_id doesn't exist
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// # use janet_ai_retriever::retrieval::file_index::ChunkRef;
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    /// let model_id = "fastembed:sentence-transformers/all-MiniLM-L6-v2:latest:384:norm";
    ///
    /// let chunks: Vec<ChunkRef> = vec![/* chunks with embeddings */];
    /// index.upsert_chunks_with_model(&chunks, model_id).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn upsert_chunks_with_model(
        &self,
        chunks: &[ChunkRef],
        model_id: &str,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        for chunk in chunks {
            let embedding_bytes = chunk
                .embedding
                .as_ref()
                .map(|e| bytemuck::cast_slice::<half::f16, u8>(e));

            sqlx::query(
                r#"
                INSERT INTO chunks (file_hash, relative_path, line_start, line_end, content, embedding, model_id)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ON CONFLICT(file_hash, line_start, line_end) DO UPDATE SET
                    relative_path = excluded.relative_path,
                    content = excluded.content,
                    embedding = excluded.embedding,
                    model_id = excluded.model_id
                "#,
            )
            .bind(&chunk.file_hash[..])
            .bind(&chunk.relative_path)
            .bind(chunk.line_start as i64)
            .bind(chunk.line_end as i64)
            .bind(&chunk.content)
            .bind(embedding_bytes)
            .bind(model_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Get chunks by model ID
    /// Get all chunks associated with a specific embedding model.
    ///
    /// Retrieves all stored chunks that were generated using the specified
    /// embedding model, including their content and embeddings.
    ///
    /// # Arguments
    /// * `model_id` - ID of the embedding model to filter by
    ///
    /// # Returns
    /// Vector of chunk references associated with the model
    ///
    /// # Errors
    /// - Database query errors
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    /// let model_id = "fastembed:sentence-transformers/all-MiniLM-L6-v2:latest:384:norm";
    ///
    /// let chunks = index.get_chunks_by_model(model_id).await?;
    /// println!("Found {} chunks for model {}", chunks.len(), model_id);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_chunks_by_model(&self, model_id: &str) -> Result<Vec<ChunkRef>> {
        let rows = sqlx::query(
            "SELECT id, file_hash, relative_path, line_start, line_end, content, embedding
             FROM chunks WHERE model_id = ?1 ORDER BY relative_path, line_start",
        )
        .bind(model_id)
        .fetch_all(&self.pool)
        .await?;

        let mut chunks = Vec::new();
        for row in rows {
            let id: i64 = row.get("id");
            let file_hash_bytes: Vec<u8> = row.get("file_hash");
            let relative_path: String = row.get("relative_path");
            let line_start: i64 = row.get("line_start");
            let line_end: i64 = row.get("line_end");
            let content: String = row.get("content");
            let embedding_bytes: Option<Vec<u8>> = row.get("embedding");

            let mut file_hash = [0u8; 32];
            file_hash.copy_from_slice(&file_hash_bytes[..32]);

            let embedding =
                embedding_bytes.map(|bytes| bytemuck::cast_slice::<u8, half::f16>(&bytes).to_vec());

            chunks.push(ChunkRef {
                id: Some(id),
                file_hash,
                relative_path,
                line_start: line_start as usize,
                line_end: line_end as usize,
                content,
                embedding,
            });
        }

        Ok(chunks)
    }

    /// Check if a new embedding model is compatible with existing data.
    ///
    /// This checks whether the provided model is compatible with embeddings already
    /// stored in the index. Models are considered compatible if they have the same
    /// dimensions and normalization settings.
    ///
    /// # Arguments
    /// * `current_model` - Model metadata to check for compatibility
    ///
    /// # Returns
    /// `true` if the model is compatible with existing embeddings, `false` otherwise
    ///
    /// # Errors
    /// - Database query errors
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::{EnhancedFileIndex, EmbeddingModelMetadata};
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    ///
    /// let new_model = EmbeddingModelMetadata::new(
    ///     "different-model".to_string(),
    ///     "fastembed".to_string(),
    ///     384
    /// );
    ///
    /// if !index.check_model_compatibility(&new_model).await? {
    ///     println!("Model incompatible - will need to re-embed existing data");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn check_model_compatibility(
        &self,
        current_model: &EmbeddingModelMetadata,
    ) -> Result<bool> {
        if let Some(existing_model) = self.get_embedding_model(&current_model.model_id()).await? {
            Ok(existing_model.dimension == current_model.dimension
                && existing_model.provider == current_model.provider
                && existing_model.normalized == current_model.normalized)
        } else {
            // No existing model, so it's compatible
            Ok(true)
        }
    }

    /// Delete chunks by model ID (useful when changing embedding models)
    /// Delete all chunks associated with a specific embedding model.
    ///
    /// This removes all chunks that were generated using the specified embedding model.
    /// Useful when switching to a new embedding model or cleaning up old data.
    ///
    /// # Arguments
    /// * `model_id` - ID of the embedding model whose chunks should be deleted
    ///
    /// # Returns
    /// Number of chunks that were deleted
    ///
    /// # Errors
    /// - Database query errors
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    /// let old_model_id = "fastembed:old-model:latest:256:norm";
    ///
    /// let deleted_count = index.delete_chunks_by_model(old_model_id).await?;
    /// println!("Deleted {} chunks from old model", deleted_count);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn delete_chunks_by_model(&self, model_id: &str) -> Result<usize> {
        let result = sqlx::query("DELETE FROM chunks WHERE model_id = ?1")
            .bind(model_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() as usize)
    }

    /// Search for similar chunks using embedding similarity.
    ///
    /// This performs a semantic similarity search by comparing the query embedding
    /// against all stored chunk embeddings using cosine similarity. Results are
    /// ranked by similarity score (highest first).
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector to search for similar chunks
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// Vector of chunk references ordered by similarity (most similar first)
    ///
    /// # Errors
    /// - Database query errors
    /// - Embedding deserialization errors
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    ///
    /// // Create a query embedding (normally from your embedding model)
    /// let query_embedding: Vec<half::f16> = vec![half::f16::from_f32(0.1); 384];
    ///
    /// let similar_chunks = index.search_similar_chunks(&query_embedding, 10).await?;
    /// for chunk in similar_chunks {
    ///     println!("Similar: {}", chunk.content);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_similar_chunks(
        &self,
        query_embedding: &[half::f16],
        limit: usize,
    ) -> Result<Vec<ChunkRef>> {
        // For now, implement a simple cosine similarity search in Rust
        // In a production system, you might want to use a vector database

        let rows = sqlx::query(
            "SELECT id, file_hash, relative_path, line_start, line_end, content, embedding
             FROM chunks WHERE embedding IS NOT NULL ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut similarities: Vec<(f32, ChunkRef)> = Vec::new();

        for row in rows {
            let id: i64 = row.get("id");
            let file_hash_bytes: Vec<u8> = row.get("file_hash");
            let relative_path: String = row.get("relative_path");
            let line_start: i64 = row.get("line_start");
            let line_end: i64 = row.get("line_end");
            let content: String = row.get("content");
            let embedding_bytes: Option<Vec<u8>> = row.get("embedding");

            if let Some(embedding_bytes) = embedding_bytes {
                let chunk_embedding: Vec<half::f16> =
                    bytemuck::cast_slice::<u8, half::f16>(&embedding_bytes).to_vec();

                // Calculate cosine similarity
                let similarity = calculate_cosine_similarity(query_embedding, &chunk_embedding);

                let mut file_hash = [0u8; 32];
                file_hash.copy_from_slice(&file_hash_bytes[..32]);

                let chunk_ref = ChunkRef {
                    id: Some(id),
                    file_hash,
                    relative_path,
                    line_start: line_start as usize,
                    line_end: line_end as usize,
                    content,
                    embedding: Some(chunk_embedding),
                };

                similarities.push((similarity, chunk_ref));
            }
        }

        // Sort by similarity (descending) and take the top results
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);

        Ok(similarities.into_iter().map(|(_, chunk)| chunk).collect())
    }

    /// Get comprehensive statistics about the stored index.
    ///
    /// Returns detailed statistics about the content stored in the database,
    /// including counts of files, chunks, and embeddings. This provides insight
    /// into the size and completeness of the indexed data.
    ///
    /// # Returns
    /// IndexStats containing counts and statistics about the indexed data
    ///
    /// # Errors
    /// - Database query errors
    ///
    /// # Example
    /// ```no_run
    /// # use janet_ai_retriever::retrieval::enhanced_index::EnhancedFileIndex;
    /// # use std::path::Path;
    /// # async fn example() -> anyhow::Result<()> {
    /// let index = EnhancedFileIndex::open(Path::new(".")).await?;
    ///
    /// let stats = index.get_index_stats().await?;
    /// println!("Index contains {} files with {} chunks",
    ///          stats.files_count, stats.chunks_count);
    /// println!("{} chunks have embeddings", stats.embedded_chunks_count);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_index_stats(&self) -> Result<IndexStats> {
        let files_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM files")
            .fetch_one(&self.pool)
            .await?;

        let chunks_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks")
            .fetch_one(&self.pool)
            .await?;

        let embeddings_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
                .fetch_one(&self.pool)
                .await?;

        let models_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM embedding_models")
            .fetch_one(&self.pool)
            .await?;

        Ok(IndexStats {
            files_count: files_count as usize,
            chunks_count: chunks_count as usize,
            embeddings_count: embeddings_count as usize,
            models_count: models_count as usize,
        })
    }
}

/// Calculate cosine similarity between two f16 embedding vectors
fn calculate_cosine_similarity(a: &[half::f16], b: &[half::f16]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| f32::from(*x) * f32::from(*y))
        .sum();

    let norm_a: f32 = a.iter().map(|x| f32::from(*x).powi(2)).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| f32::from(*x).powi(2)).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Statistics about the index
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub files_count: usize,
    pub chunks_count: usize,
    pub embeddings_count: usize,
    pub models_count: usize,
}

// Implement Deref to allow transparent access to FileIndex methods
impl std::ops::Deref for EnhancedFileIndex {
    type Target = FileIndex;

    fn deref(&self) -> &Self::Target {
        &self.file_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_enhanced_index_creation() -> Result<()> {
        let temp_dir = tempdir()?;
        let index = EnhancedFileIndex::open_memory(temp_dir.path()).await?;

        // Should be able to access underlying FileIndex methods
        let stats = index.get_index_stats().await?;
        assert_eq!(stats.files_count, 0);
        assert_eq!(stats.chunks_count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_embedding_model_registration() -> Result<()> {
        let temp_dir = tempdir()?;
        let index = EnhancedFileIndex::open_memory(temp_dir.path()).await?;

        let model =
            EmbeddingModelMetadata::new("test-model".to_string(), "test-provider".to_string(), 384)
                .with_normalized(true)
                .with_config("batch_size".to_string(), "16".to_string());

        index.register_embedding_model(&model).await?;

        let retrieved = index.get_embedding_model(&model.model_id()).await?;
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.model_name, "test-model");
        assert_eq!(retrieved.dimension, 384);
        assert!(retrieved.normalized);

        Ok(())
    }

    #[tokio::test]
    async fn test_index_metadata() -> Result<()> {
        let temp_dir = tempdir()?;
        let index = EnhancedFileIndex::open_memory(temp_dir.path()).await?;

        let metadata = IndexMetadata::new("test-repo".to_string())
            .with_metadata("test_key".to_string(), "test_value".to_string());

        index.upsert_index_metadata(&metadata).await?;

        let retrieved = index.get_index_metadata("test-repo").await?;
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.repository, "test-repo");
        assert_eq!(
            retrieved.metadata.get("test_key"),
            Some(&"test_value".to_string())
        );

        Ok(())
    }
}
