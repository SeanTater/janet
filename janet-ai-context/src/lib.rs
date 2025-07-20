pub mod text;

// Re-export the main chunking function for external use
pub use text::{create_builder_for_path, get_delimiters_for_path, TextContextBuilder, TextChunk};
