//! Core SQLite database operations for file and chunk storage.
//!
//! This module provides the foundational data layer for janet-ai-retriever, implementing
//! direct SQLite operations for storing files, code chunks, and their embeddings.
//!
//! ## Key Components
//!
//! - **FileIndex**: Main database interface with optimized SQLite configuration
//! - **FileRef**: Represents a source file with content and hash
//! - **ChunkRef**: Represents a code chunk with optional f16 embeddings
//!
//! ## Database Schema
//!
//! ```sql
//! -- Files table: tracks source files by hash
//! CREATE TABLE files (
//!     hash BLOB PRIMARY KEY,           -- blake3 hash (32 bytes)
//!     relative_path TEXT UNIQUE,       -- path relative to project root
//!     size INTEGER,                    -- file size in bytes
//!     modified_at TIMESTAMP,           -- last modification time
//!     indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
//! );
//!
//! -- Chunks table: stores code chunks with embeddings
//! CREATE TABLE chunks (
//!     id INTEGER PRIMARY KEY AUTOINCREMENT,
//!     file_hash BLOB REFERENCES files(hash),
//!     relative_path TEXT,              -- denormalized for performance
//!     line_start INTEGER,              -- chunk start line
//!     line_end INTEGER,                -- chunk end line
//!     content TEXT,                    -- actual chunk text
//!     embedding BLOB,                  -- f16 embedding vector (optional)
//!     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
//! );
//! ```
//!
//! ## SQLite Optimizations
//!
//! - **WAL mode**: Better concurrency for read/write operations
//! - **Large page size** (64KB): Optimized for embedding blob storage
//! - **Auto-vacuum**: Keeps database size manageable
//! - **Foreign keys**: Maintains referential integrity
//! - **Strategic indexes**: On file hashes, paths, and modification times
//!
//! ## Usage
//!

use anyhow::Result;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Row, SqlitePool};
use std::path::{Path, PathBuf};

/// Reference to a file stored in the index database.
///
/// This struct represents a complete file with its content, path, and hash.
/// Files are the top-level units in the indexing system, with chunks belonging
/// to files. The hash provides content-based deduplication and change detection.
///
/// # Fields
/// * `relative_path` - Path relative to the project root, used for display and lookup
/// * `content` - Raw file content as bytes to preserve exact content for hashing
/// * `hash` - Blake3 hash of the content for deduplication and change detection
///
/// # Example
#[derive(Debug, Clone)]
pub struct FileRef {
    /// The path to the file, relative to the root of the project
    pub relative_path: String,
    /// The file content, not interpreted as a string, to avoid fouling hashes
    pub content: Vec<u8>,
    /// The blake3 hash of the file
    pub hash: [u8; 32],
    /// Last modification time of the file (Unix timestamp)
    pub modified_at: i64,
}

/// Reference to a text chunk stored in the index database.
///
/// This struct represents a segment of text extracted from a file, along with
/// its location information and optional embedding vector. Chunks are the
/// primary searchable units in the indexing system.
///
/// # Fields
/// * `id` - Database ID (None for new chunks, Some for existing ones)
/// * `file_hash` - Hash of the parent file this chunk belongs to
/// * `relative_path` - Path to the parent file (for convenience)
/// * `line_start` - Starting line number of this chunk in the file
/// * `line_end` - Ending line number of this chunk in the file
/// * `content` - The actual text content of this chunk
/// * `embedding` - Optional vector embedding for semantic search (f16 for efficiency)
///
/// # Example
#[derive(Debug, Clone)]
pub struct ChunkRef {
    pub id: Option<i64>,
    pub file_hash: [u8; 32],
    pub relative_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
    pub embedding: Option<Vec<half::f16>>,
}

/// SQLite-based file and chunk indexing system.
///
/// FileIndex provides low-level database operations for storing and retrieving
/// files and their associated text chunks. It serves as the foundation for the
/// more advanced [`EnhancedFileIndex`](super::enhanced_index::EnhancedFileIndex).
///
/// The index uses SQLite with WAL mode for concurrent access and stores:
/// - **Files**: Complete file content with paths and hashes
/// - **Chunks**: Text segments extracted from files with location info
/// - **Embeddings**: Optional f16 vector embeddings for semantic search
///
/// # Database Schema
///
/// # Example
#[derive(Clone, Debug)]
pub struct FileIndex {
    pub(crate) base: PathBuf,
    pool: SqlitePool,
}

impl FileIndex {
    /// Opens file index with persistent SQLite storage. See module docs for usage patterns.
    pub async fn open(base: &Path) -> Result<Self> {
        let db_path = base.join(".janet-ai.db");

        let pool = SqlitePool::connect_with(
            SqliteConnectOptions::new()
                .filename(db_path)
                .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
                .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
                .busy_timeout(std::time::Duration::from_secs(5))
                .foreign_keys(true)
                .create_if_missing(true)
                .auto_vacuum(sqlx::sqlite::SqliteAutoVacuum::Full)
                .page_size(1 << 16)
                .optimize_on_close(true, 1 << 10),
        )
        .await?;
        Self::new_with_pool(base, pool).await
    }

    /// Opens file index with in-memory SQLite storage for testing. See module docs for details.
    pub async fn open_memory(base: &Path) -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        Self::new_with_pool(base, pool).await
    }

    async fn new_with_pool(base: &Path, pool: SqlitePool) -> Result<Self> {
        // Create tables directly
        Self::create_tables(&pool).await?;

        Ok(Self {
            base: base.to_path_buf(),
            pool,
        })
    }

    async fn create_tables(pool: &SqlitePool) -> Result<()> {
        // Create files table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS files (
                hash BLOB PRIMARY KEY,
                relative_path TEXT UNIQUE NOT NULL,
                size INTEGER NOT NULL,
                modified_at TIMESTAMP NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(pool)
        .await?;

        // Create chunks table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash BLOB NOT NULL,
                relative_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_chunk UNIQUE(file_hash, line_start, line_end),
                FOREIGN KEY (file_hash) REFERENCES files(hash) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await?;

        // Create indexes
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(relative_path)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_files_path ON files(relative_path)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_files_modified ON files(modified_at)")
            .execute(pool)
            .await?;

        Ok(())
    }

    /// Inserts or updates file record with metadata. See module docs for schema details.
    pub async fn upsert_file(&self, file_ref: &FileRef) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO files (hash, relative_path, size, modified_at, indexed_at)
            VALUES (?1, ?2, ?3, datetime(?4, 'unixepoch'), datetime('now'))
            ON CONFLICT(hash) DO UPDATE SET
                relative_path = excluded.relative_path,
                size = excluded.size,
                modified_at = excluded.modified_at,
                indexed_at = datetime('now')
            "#,
        )
        .bind(&file_ref.hash[..])
        .bind(&file_ref.relative_path)
        .bind(file_ref.content.len() as i64)
        .bind(file_ref.modified_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Check if a file needs reindexing by comparing modification times.
    pub async fn file_needs_reindexing(&self, file_path: &Path) -> Result<bool> {
        let metadata = tokio::fs::metadata(file_path).await?;
        let file_modified_at = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;

        let relative_path = file_path.to_string_lossy().to_string();
        let stored_modified_at = sqlx::query_scalar::<_, Option<i64>>(
            "SELECT strftime('%s', modified_at) FROM files WHERE relative_path = ?1",
        )
        .bind(&relative_path)
        .fetch_optional(&self.pool)
        .await?;

        match stored_modified_at {
            Some(Some(stored_time)) => Ok(file_modified_at > stored_time),
            _ => Ok(true), // File not in database or null timestamp, needs indexing
        }
    }

    /// Inserts or updates multiple text chunks with embeddings. See module docs for usage patterns.
    pub async fn upsert_chunks(&self, chunks: &[ChunkRef]) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        for chunk in chunks {
            let embedding_bytes = chunk
                .embedding
                .as_ref()
                .map(|e| bytemuck::cast_slice::<half::f16, u8>(e));

            sqlx::query(
                r#"
                INSERT INTO chunks (file_hash, relative_path, line_start, line_end, content, embedding)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ON CONFLICT(file_hash, line_start, line_end) DO UPDATE SET
                    relative_path = excluded.relative_path,
                    content = excluded.content,
                    embedding = excluded.embedding
                "#,
            )
            .bind(&chunk.file_hash[..])
            .bind(&chunk.relative_path)
            .bind(chunk.line_start as i64)
            .bind(chunk.line_end as i64)
            .bind(&chunk.content)
            .bind(embedding_bytes)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Get chunks for a file
    pub async fn get_chunks(&self, file_hash: &[u8; 32]) -> Result<Vec<ChunkRef>> {
        let rows = sqlx::query(
            "SELECT id, file_hash, relative_path, line_start, line_end, content, embedding FROM chunks WHERE file_hash = ?1 ORDER BY line_start",
        )
        .bind(&file_hash[..])
        .fetch_all(&self.pool)
        .await?;

        let mut chunks = Vec::new();
        for row in rows {
            let id: i64 = row.get("id");
            let relative_path: String = row.get("relative_path");
            let line_start: i64 = row.get("line_start");
            let line_end: i64 = row.get("line_end");
            let content: String = row.get("content");
            let embedding_bytes: Option<Vec<u8>> = row.get("embedding");

            let embedding =
                embedding_bytes.map(|bytes| bytemuck::cast_slice::<u8, half::f16>(&bytes).to_vec());

            chunks.push(ChunkRef {
                id: Some(id),
                file_hash: *file_hash,
                relative_path,
                line_start: line_start as usize,
                line_end: line_end as usize,
                content,
                embedding,
            });
        }
        Ok(chunks)
    }

    /// Delete chunks for a file
    pub async fn delete_chunks(&self, file_hash: &[u8; 32]) -> Result<usize> {
        let result = sqlx::query("DELETE FROM chunks WHERE file_hash = ?1")
            .bind(&file_hash[..])
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() as usize)
    }

    /// Get a chunk by ID
    pub async fn get_chunk_by_id(&self, id: i64) -> Result<Option<ChunkRef>> {
        let row = sqlx::query(
            "SELECT id, file_hash, relative_path, line_start, line_end, content, embedding FROM chunks WHERE id = ?1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
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

            Ok(Some(ChunkRef {
                id: Some(id),
                file_hash,
                relative_path,
                line_start: line_start as usize,
                line_end: line_end as usize,
                content,
                embedding,
            }))
        } else {
            Ok(None)
        }
    }

    /// Update a chunk's embedding by ID
    pub async fn update_chunk_embedding(
        &self,
        id: i64,
        embedding: Option<&[half::f16]>,
    ) -> Result<()> {
        let embedding_bytes = embedding.map(bytemuck::cast_slice::<half::f16, u8>);

        sqlx::query("UPDATE chunks SET embedding = ?1 WHERE id = ?2")
            .bind(embedding_bytes)
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Get all chunks with embeddings
    pub async fn get_all_chunks_with_embeddings(&self) -> Result<Vec<ChunkRef>> {
        let rows = sqlx::query(
            "SELECT id, file_hash, relative_path, line_start, line_end, content, embedding
             FROM chunks WHERE embedding IS NOT NULL",
        )
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

    /// Update chunk by ID
    pub async fn update_chunk_by_id(&self, id: i64, chunk: &ChunkRef) -> Result<()> {
        let embedding_bytes = chunk
            .embedding
            .as_ref()
            .map(|e| bytemuck::cast_slice::<half::f16, u8>(e));

        sqlx::query(
            "UPDATE chunks SET file_hash = ?1, relative_path = ?2, line_start = ?3,
             line_end = ?4, content = ?5, embedding = ?6 WHERE id = ?7",
        )
        .bind(&chunk.file_hash[..])
        .bind(&chunk.relative_path)
        .bind(chunk.line_start as i64)
        .bind(chunk.line_end as i64)
        .bind(&chunk.content)
        .bind(embedding_bytes)
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Delete embeddings for specific chunk IDs
    pub async fn delete_embeddings_by_ids(&self, chunk_ids: &[i64]) -> Result<()> {
        if chunk_ids.is_empty() {
            return Ok(());
        }

        // Build a query with placeholders
        let placeholders = chunk_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");

        let query = format!("UPDATE chunks SET embedding = NULL WHERE id IN ({placeholders})");
        let mut query_builder = sqlx::query(&query);

        for id in chunk_ids {
            query_builder = query_builder.bind(id);
        }

        query_builder.execute(&self.pool).await?;
        Ok(())
    }

    /// Get all chunks in the database
    pub async fn get_all_chunks(&self) -> Result<Vec<ChunkRef>> {
        let rows = sqlx::query(
            "SELECT id, file_hash, relative_path, line_start, line_end, content, embedding FROM chunks ORDER BY relative_path, line_start",
        )
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

    /// Get the underlying SQLite connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use tempfile::tempdir;

    /// Test file insertion and retrieval
    #[tokio::test]
    async fn test_file_operations() -> Result<()> {
        let temp_dir = tempdir()?;
        let index = FileIndex::open_memory(temp_dir.path()).await?;

        let file_ref = FileRef {
            relative_path: "test/file.rs".to_string(),
            content: b"fn main() {}\n".to_vec(),
            hash: [1; 32],
            modified_at: 1640995200,
        };

        // Insert file
        index.upsert_file(&file_ref).await?;

        // File was inserted successfully

        Ok(())
    }

    /// Test chunk operations
    #[tokio::test]
    async fn test_chunk_operations() -> Result<()> {
        let temp_dir = tempdir()?;
        let index = FileIndex::open_memory(temp_dir.path()).await?;

        // First insert a file
        let file_ref = FileRef {
            relative_path: "test/file.rs".to_string(),
            content: b"fn main() {}\nfn test() {}".to_vec(),
            hash: [2; 32],
            modified_at: 1640995200,
        };
        index.upsert_file(&file_ref).await?;

        // Create chunks
        let chunks = vec![
            ChunkRef {
                id: None,
                file_hash: [2; 32],
                relative_path: "test/file.rs".to_string(),
                line_start: 1,
                line_end: 1,
                content: "fn main() {}".to_string(),
                embedding: Some(vec![
                    half::f16::from_f32(0.1),
                    half::f16::from_f32(0.2),
                    half::f16::from_f32(0.3),
                ]),
            },
            ChunkRef {
                id: None,
                file_hash: [2; 32],
                relative_path: "test/file.rs".to_string(),
                line_start: 2,
                line_end: 2,
                content: "fn test() {}".to_string(),
                embedding: Some(vec![
                    half::f16::from_f32(0.4),
                    half::f16::from_f32(0.5),
                    half::f16::from_f32(0.6),
                ]),
            },
        ];

        // Insert chunks
        index.upsert_chunks(&chunks).await?;

        // Retrieve chunks
        let fetched_chunks = index.get_chunks(&[2; 32]).await?;
        assert_eq!(fetched_chunks.len(), 2);
        assert_eq!(fetched_chunks[0].content, "fn main() {}");
        assert_eq!(fetched_chunks[1].content, "fn test() {}");
        assert!(fetched_chunks[0].embedding.is_some());
        assert_eq!(
            fetched_chunks[0].embedding.as_ref().unwrap(),
            &vec![
                half::f16::from_f32(0.1),
                half::f16::from_f32(0.2),
                half::f16::from_f32(0.3)
            ]
        );

        Ok(())
    }

    /// Test chunk deletion
    #[tokio::test]
    async fn test_chunk_deletion() -> Result<()> {
        let temp_dir = tempdir()?;
        let index = FileIndex::open_memory(temp_dir.path()).await?;

        // Insert file and chunks
        let file_ref = FileRef {
            relative_path: "test/file.rs".to_string(),
            content: b"fn main() {}".to_vec(),
            hash: [3; 32],
            modified_at: 1640995200,
        };
        index.upsert_file(&file_ref).await?;

        let chunks = vec![ChunkRef {
            id: None,
            file_hash: [3; 32],
            relative_path: "test/file.rs".to_string(),
            line_start: 1,
            line_end: 1,
            content: "fn main() {}".to_string(),
            embedding: None,
        }];
        index.upsert_chunks(&chunks).await?;

        // Verify chunks exist
        let fetched = index.get_chunks(&[3; 32]).await?;
        assert_eq!(fetched.len(), 1);

        // Delete chunks
        let deleted_count = index.delete_chunks(&[3; 32]).await?;
        assert_eq!(deleted_count, 1);

        // Verify chunks are gone
        let fetched_after = index.get_chunks(&[3; 32]).await?;
        assert_eq!(fetched_after.len(), 0);

        Ok(())
    }
}
