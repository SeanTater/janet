//! Error types for the embedding system

use std::path::PathBuf;

/// Result type for embedding operations
pub type Result<T> = std::result::Result<T, EmbedError>;

/// Comprehensive error type for embedding operations
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// Error when model files are not found or invalid
    #[error("Model file not found: {path}")]
    ModelFileNotFound { path: PathBuf },

    /// Error when model configuration is invalid
    #[error("Invalid model configuration: {message}")]
    InvalidConfig { message: String },

    /// Error during model initialization
    #[error("Model initialization failed: {source}")]
    ModelInitialization {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Error during embedding generation
    #[error("Embedding generation failed: {source}")]
    EmbeddingGeneration {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// IO errors when reading model files
    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    /// Async task join errors
    #[error("Async task failed: {source}")]
    AsyncTask {
        #[from]
        source: tokio::task::JoinError,
    },

    /// Generic errors from other libraries
    #[error("External error: {source}")]
    External {
        #[from]
        source: anyhow::Error,
    },
}

impl EmbedError {
    /// Create a model initialization error
    pub fn model_init<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::ModelInitialization {
            source: Box::new(source),
        }
    }

    /// Create an embedding generation error
    pub fn embedding_gen<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::EmbeddingGeneration {
            source: Box::new(source),
        }
    }

    /// Create an invalid config error
    pub fn invalid_config<S: Into<String>>(message: S) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }
}