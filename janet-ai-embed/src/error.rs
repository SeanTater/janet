//! Error types for the embedding system

use std::path::PathBuf;

/// Result type for embedding operations.
///
/// This is a convenience type alias that uses [`EmbedError`] as the error type.
/// Used throughout the crate for operations that can fail.
///
/// # Example
pub type Result<T> = std::result::Result<T, EmbedError>;

/// Comprehensive error type for all embedding operations.
///
/// This enum covers all possible error conditions that can occur when working
/// with embedding models, from configuration errors to runtime failures during
/// embedding generation. Each variant provides specific context about the failure.
///
/// The error type integrates with the [`thiserror`] crate for automatic
/// [`std::error::Error`] implementation and supports error chaining for
/// detailed error context.
///
/// # Error Categories
///
/// - **Configuration Errors**: Invalid model settings or missing files
/// - **Initialization Errors**: Failures during model loading or setup
/// - **Runtime Errors**: Problems during actual embedding generation
/// - **IO Errors**: File system access issues
/// - **External Errors**: Failures from dependencies
///
/// # Example
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
    /// Create a model initialization error from any error type.
    ///
    /// This is a convenience constructor for wrapping errors that occur during
    /// model loading or initialization into the appropriate variant.
    ///
    /// # Arguments
    /// * `source` - The underlying error that caused the initialization failure
    ///
    /// # Returns
    /// A new [`EmbedError::ModelInitialization`] variant
    ///
    /// # Example
    pub fn model_init<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::ModelInitialization {
            source: Box::new(source),
        }
    }

    /// Create an embedding generation error from any error type.
    ///
    /// This is a convenience constructor for wrapping errors that occur during
    /// the actual embedding generation process.
    ///
    /// # Arguments
    /// * `source` - The underlying error that caused the embedding generation failure
    ///
    /// # Returns
    /// A new [`EmbedError::EmbeddingGeneration`] variant
    ///
    /// # Example
    pub fn embedding_gen<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::EmbeddingGeneration {
            source: Box::new(source),
        }
    }

    /// Create an invalid configuration error with a custom message.
    ///
    /// This is a convenience constructor for configuration validation errors,
    /// such as invalid model parameters, missing required settings, or
    /// incompatible configuration combinations.
    ///
    /// # Arguments
    /// * `message` - A descriptive error message explaining what's wrong with the configuration
    ///
    /// # Returns
    /// A new [`EmbedError::InvalidConfig`] variant
    ///
    /// # Example
    pub fn invalid_config<S: Into<String>>(message: S) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }
}
