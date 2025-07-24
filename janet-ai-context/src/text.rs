//! This module provides utilities for building text contexts for retrieval models,
//! specifically designed for use within a RAG (Retrieval Augmented Generation) system.
//!
//! The primary goal is to transform raw code content into structured "passages"
//! that can be effectively used by models like BERT for retrieval tasks.
//! These passages include metadata about the code's origin (repository, path)
//! and are segmented into manageable chunks to optimize for model input limits.
//!
//! The module defines two main structs:
//! - [`TextContextBuilder`]: Responsible for configuring how text is chunked and
//!   for generating the final context string.
//! - [`TextChunk`]: Represents a single segment of text extracted from a file,
//!   along with its associated metadata.
//!
//! # Example Context Format
//!
//! The `TextContextBuilder` aims to produce code snippets that look like this:
//!
//! ```rs
//! passage: {"repo": "example", "path": "the_crate/src/lib.rs"}
//!
//! context: {"module": "self"}
//! /// Create a wrapped database connection
//! use std::sync::{Arc, Mutex};
//! use anyhow::Result;
//! use some_crate::module::DBConnection;
//! use other_crate::stuff;
//! use crate::sync::DatabaseHandle;
//! // ... (more code)
//!
//! context: {"module": "crate::sync"}
//! struct DatabaseHandle {
//!     thing: Arc<Mutex<DBConnection>>,
//!     // ... (more fields)
//! }
//!
//! focus: {}
//! pub fn connect(host: str) -> Result<DatabaseHandle> {
//!     DatabaseHandle {
//!         thing: Arc::new(Mutex::new(DBConnection::connect(host)?))
//!     }
//! }
//!
//! ```
//!
//! # Key Features
//!
//! *   **Metadata Inclusion**: Each passage starts with a "document" or "passage" prefix
//!     followed by JSON containing repository and file path information.
//! *   **Contextual Segmentation**: Code is split into segments based on configurable
//!     delimiters (e.g., newlines, double newlines, spaces) to create meaningful chunks.
//! *   **Recursive Splitting**: The `split_recursively_into_segments` method intelligently
//!     breaks down text, prioritizing larger delimiters first, ensuring that chunks
//!     do not exceed a specified maximum size.
//! *   **Chunk Reconstruction**: The `get_chunks` method ensures that all original
//!     file content can be reconstructed by concatenating the generated `TextChunk`s.
//!
//! # Usage
//!
//! To use this module, you typically:
//! 1.  Create a `TextContextBuilder` instance with the repository name, file path,
//!     and a set of regular expression patterns to use as delimiters.
//! 2.  Call `get_chunks` to obtain a vector of `TextChunk`s from your file content.
//! 3.  (Optional) Use the `build` method to format the entire file content into a
//!     single passage string, primarily for file-level context.
//!
//! ```
//! use janet_ai_context::text::{TextContextBuilder, TextChunk};
//!
//! let repo_name = "my_rust_project".to_string();
//! let file_path = "src/main.rs".to_string();
//! // Delimiters are used to split the text into logical segments.
//! // They are applied in order: try splitting by double newline, then single newline, then space.
//! let delimiter_patterns = vec![r"\n\n", r"\n", r" "];
//! let max_chunk_length = 500;
//!
//! let builder = TextContextBuilder::new(repo_name, file_path, &delimiter_patterns, max_chunk_length);
//!
//! let file_content = r#"
//! fn main() {
//!     println!("Hello, world!");
//!
//!     // This is another part of the code.
//!     let x = 10;
//!     let y = 20;
//!     println!("Sum: {}", x + y);
//! }
//! "#;
//!
//! let chunks = builder.get_chunks(file_content);
//!
//! // Verify that chunks were created and their content is correct
//! assert!(!chunks.is_empty());
//! let reconstructed_content: String = chunks.iter().map(|c| c.chunk_text.as_str()).collect();
//! assert_eq!(reconstructed_content, file_content);
//!
//! // You can iterate over chunks to process them individually
//! for chunk in chunks {
//!     println!("Chunk sequence: {}", chunk.sequence);
//!     println!("Chunk text: \"{}\"", &chunk.chunk_text);
//!     // Further processing of each chunk can happen here
//! }
//! ```
use regex::Regex;
use serde::{Serialize, Serializer};
use std::ops::Range;
use std::path::Path;

/// Default regular expression patterns for splitting Markdown content into logical segments.
///
/// These delimiters are ordered from most significant to least significant,
/// guiding the recursive splitting process in `TextContextBuilder::get_chunks`.
///
/// - `^\s*#{1,6}\s+.*$`: Matches Markdown headings (e.g., `# Heading`, `## Subheading`).
/// - ```` ``` ````: Matches Markdown code block fences.
/// - `\n\n`: Splits by double newlines, typically separating paragraphs.
/// - `^\s*[-*+]\s+`: Matches unordered list item markers (e.g., `- item`, `* item`).
/// - `^\s*\d+\.\s+`: Matches ordered list item markers (e.g., `1. item`).
/// - `\n`: Splits by single newlines, useful for breaking long lines within paragraphs.
/// - ` `: Splits by spaces, the most granular delimiter.
pub const DEFAULT_MARKDOWN_DELIMITERS: &[&str] = &[
    r"^\s*#{1,6}\s+.*$", // Headings (e.g., # Heading, ## Subheading)
    r"```",              // Code block fences
    r"\n\n",             // Paragraphs
    r"^\s*[-*+]\s+",     // Unordered list items (e.g., - item, * item)
    r"^\s*\d+\.\s+",     // Ordered list items (e.g., 1. item)
    r"\n",               // Line breaks
    r" ",                // Spaces
];

/// Code-specific delimiters for better chunking of programming languages
pub const CODE_DELIMITERS: &[&str] = &[
    r"(?m)^(pub\s+)?(struct|enum|trait|impl|fn|class|def|function|interface)\s+\w+", // Declarations
    r"(?m)^(import|use|from|#include)\s+", // Import statements
    r"(?m)^(package|namespace|module)\s+", // Module declarations
    r"\n\n",                               // Paragraph breaks
    r"\n",                                 // Line breaks
    r" ",                                  // Spaces
];

/// Get appropriate text delimiters based on file extension.
///
/// This function analyzes the file path and returns delimiters suitable for chunking
/// that type of content. Different file types have different natural breaking points:
/// - Code files use function/class boundaries
/// - Markdown files use headers and paragraphs
/// - Documentation files use documentation-specific delimiters
///
/// # Arguments
/// * `path` - The file path to analyze
///
/// # Returns
/// A static slice of delimiter strings, ordered by preference (most important first)
///
/// # Examples
/// ```
/// use std::path::Path;
/// use janet_ai_context::get_delimiters_for_path;
///
/// let rust_delims = get_delimiters_for_path(Path::new("src/main.rs"));
/// // Returns code delimiters like ["\nfn ", "\nstruct ", "\nimpl ", ...]
///
/// let md_delims = get_delimiters_for_path(Path::new("README.md"));
/// // Returns markdown delimiters like ["\n# ", "\n## ", "\n\n", ...]
/// ```
pub fn get_delimiters_for_path(path: &Path) -> &'static [&'static str] {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("md") | Some("markdown") | Some("txt") => DEFAULT_MARKDOWN_DELIMITERS,
        Some("rs") | Some("py") | Some("js") | Some("ts") | Some("jsx") | Some("tsx")
        | Some("go") | Some("java") | Some("c") | Some("cpp") | Some("h") | Some("hpp") => {
            CODE_DELIMITERS
        }
        _ => {
            // Check for common documentation files
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with("README") || filename.starts_with("CHANGELOG") {
                    return DEFAULT_MARKDOWN_DELIMITERS;
                }
            }
            CODE_DELIMITERS // Default to code delimiters for unknown files
        }
    }
}

/// Create a TextContextBuilder pre-configured for the given file path.
///
/// This is a convenience function that creates a builder with appropriate delimiters
/// and settings based on the file type. It automatically:
/// - Selects appropriate delimiters using [`get_delimiters_for_path`]
/// - Sets up repository and file path information
/// - Configures reasonable defaults for the file type
///
/// # Arguments
/// * `repo` - Repository name for context
/// * `file_path` - Path to the file that will be chunked
/// * `max_chunk_size` - Maximum size in characters for each chunk
///
/// # Returns
/// A configured TextContextBuilder ready for chunking
///
/// # Examples
/// ```
/// use std::path::Path;
/// use janet_ai_context::create_builder_for_path;
///
/// let builder = create_builder_for_path(
///     "my-project".to_string(),
///     Path::new("src/lib.rs"),
///     1000
/// );
///
/// let chunks = builder.get_chunks("fn main() { println!(\"Hello!\"); }");
/// ```
pub fn create_builder_for_path(
    repo: String,
    file_path: &Path,
    max_chunk_length: Option<usize>,
) -> TextContextBuilder {
    let delimiters = get_delimiters_for_path(file_path);
    let chunk_size = max_chunk_length.unwrap_or(2000);

    TextContextBuilder::new(
        repo,
        file_path.to_string_lossy().to_string(),
        delimiters,
        chunk_size,
    )
}

/// Represents a builder for creating text contexts from file content.
///
/// This struct is responsible for configuring how a given file's content
/// should be segmented into smaller, manageable chunks suitable for
/// retrieval models. It holds metadata about the repository and file path,
/// as well as a set of regular expressions used as delimiters for splitting text.
pub struct TextContextBuilder {
    repo: String,
    path: String,
    delimiters: Vec<Regex>,
    max_chunk_length: usize,
}

/// Represents a single chunk of text extracted from a file, along with its metadata.
///
/// `TextChunk` contains a portion of the original file content,
/// retaining information about its origin (repository, path), its order within
/// the sequence of chunks, and owned copies of the full file content and its own text.
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The name of the repository the file belongs to.
    pub repo: String,
    /// The path to the file within the repository.
    pub path: String,
    /// The sequence number of this chunk within the file (0-indexed).
    pub sequence: usize,
    /// The entire original file content.
    pub file_content: String,
    /// The text content of this specific chunk.
    pub chunk_text: String,
}

impl Serialize for TextChunk {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TextChunk", 5)?;
        state.serialize_field("repo", &self.repo)?;
        state.serialize_field("path", &self.path)?;
        state.serialize_field("sequence", &self.sequence)?;
        state.serialize_field("chunk_text", &self.chunk_text)?;
        state.serialize_field("summary", &self.build())?;
        state.end()
    }
}

impl TextContextBuilder {
    /// Creates a new `TextContextBuilder` instance.
    ///
    /// Initializes the builder with the repository name, file path, and a list
    /// of regular expression patterns to be used as delimiters for text splitting.
    /// The delimiters are compiled into `regex::Regex` objects.
    ///
    /// # Arguments
    ///
    /// *   `repo` - The name of the repository.
    /// *   `path` - The path to the file within the repository.
    /// *   `delimiter_patterns` - A slice of string slices, where each string is
    ///     a regular expression pattern used to split the text. Delimiters are
    ///     applied in the order they appear in the slice, from most significant
    ///     (e.g., double newline) to least significant (e.g., space).
    ///
    /// # Panics
    ///
    /// This method will panic if any of the provided `delimiter_patterns` are
    /// invalid regular expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// use janet_ai_context::text::TextContextBuilder;
    ///
    /// let repo_name = "my_project".to_string();
    /// let file_path = "src/lib.rs".to_string();
    /// // Define delimiters for splitting text.
    /// // Here, we prioritize splitting by double newlines, then single newlines, then spaces.
    /// let delimiter_patterns = vec![r"\n\n", r"\n", r" "];
    /// let max_chunk_length = 500;
    ///
    /// let builder = TextContextBuilder::new(repo_name, file_path, &delimiter_patterns, max_chunk_length);
    /// ```
    pub fn new(
        repo: String,
        path: String,
        delimiter_patterns: &[&str],
        max_chunk_length: usize,
    ) -> Self {
        let delimiters = delimiter_patterns
            .iter()
            .map(|&pattern| Regex::new(pattern).unwrap())
            .collect();

        TextContextBuilder {
            repo,
            path,
            delimiters,
            max_chunk_length,
        }
    }

    /// Create a new `TextContextBuilder` instance with default delimiters and chunk size.
    /// This is a convenience method that uses the default Markdown delimiters
    /// and a maximum chunk size of 5000 characters.
    ///
    /// # Arguments
    /// *   `repo` - The name of the repository.
    /// *   `path` - The path to the file within the repository.
    ///
    /// # Returns
    ///
    /// A new instance of `TextContextBuilder` with the specified parameters.
    pub fn with_defaults(repo: String, path: String) -> Self {
        Self::new(repo, path, DEFAULT_MARKDOWN_DELIMITERS, 5000)
    }

    /// Splits the provided `file_content` into a vector of `TextChunk`s.
    ///
    /// This method uses the configured delimiters and a maximum chunk size (500 characters)
    /// to intelligently break down the file content. It attempts to create chunks
    /// that are meaningful segments of text, respecting the delimiters, while ensuring
    /// no chunk exceeds the `max_chunk_size`.
    ///
    /// The splitting process is recursive, attempting to split by more significant
    /// delimiters first (e.g., double newlines) before falling back to less significant
    /// ones (e.g., single newlines, spaces) if a segment is still too large.
    ///
    /// # Arguments
    ///
    /// *   `file_content` - The entire content of the file as a string slice.
    ///
    /// # Returns
    ///
    /// A `Vec<TextChunk>` where each `TextChunk` represents a segment of the
    /// original `file_content`. The chunks are ordered sequentially and, when
    /// concatenated, will reconstruct the original `file_content`.
    ///
    /// # Examples
    ///
    /// ```
    /// use janet_ai_context::text::{TextContextBuilder, TextChunk};
    ///
    /// let repo_name = "example_repo".to_string();
    /// let file_path = "path/to/example.txt".to_string();
    /// let delimiter_patterns = vec![r"\n\n", r"\n", r" "];
    /// let max_chunk_length = 500;
    /// let builder = TextContextBuilder::new(repo_name.clone(), file_path.clone(), &delimiter_patterns, max_chunk_length);
    ///
    /// let file_content = "This is the first sentence. This is the second sentence.\n\nThis is a new paragraph.";
    /// let chunks = builder.get_chunks(file_content);
    ///
    /// // Verify that chunks were created
    /// assert!(!chunks.is_empty());
    ///
    /// // Verify that concatenating chunks reconstructs the original content
    /// let reconstructed_content: String = chunks.iter().map(|c| c.chunk_text.as_str()).collect();
    /// assert_eq!(reconstructed_content, file_content);
    ///
    /// // Check properties of the first chunk
    /// assert_eq!(chunks[0].repo, repo_name);
    /// assert_eq!(chunks[0].path, file_path);
    /// assert_eq!(chunks[0].sequence, 0);
    /// assert_eq!(chunks[0].file_content, file_content);
    /// assert!(chunks[0].chunk_text.len() <= max_chunk_length);
    ///
    /// // Example with content larger than max_chunk_size to demonstrate splitting
    /// let long_content = (0..100).map(|_| "A very long sentence part. ").collect::<String>();
    /// let long_chunks = builder.get_chunks(&long_content);
    /// assert!(long_chunks.len() > 1); // Should be split into multiple chunks
    /// let reconstructed_long_content: String = long_chunks.iter().map(|c| c.chunk_text.as_str()).collect();
    /// assert_eq!(reconstructed_long_content, long_content);
    /// ```
    pub fn get_chunks(&self, file_content: &str) -> Vec<TextChunk> {
        let segments = self.split_recursively_into_segments(
            file_content,
            &self.delimiters, // Use the stored delimiters
            0,                // Start with the first delimiter
            self.max_chunk_length,
            0, // Initial offset is 0
        );

        let mut chunks: Vec<TextChunk> = Vec::new();
        let mut current_chunk_start_byte = 0;
        let mut current_chunk_end_byte = 0;

        for segment_range in segments {
            let segment_text = &file_content[segment_range.clone()];

            // If adding the current segment makes the chunk too large,
            // push the current chunk and start a new one.
            if current_chunk_end_byte - current_chunk_start_byte + segment_text.len()
                > self.max_chunk_length
                && current_chunk_start_byte != current_chunk_end_byte
            {
                chunks.push(TextChunk {
                    repo: self.repo.clone(),
                    path: self.path.clone(),
                    sequence: chunks.len(),
                    file_content: file_content.to_string(),
                    chunk_text: file_content[current_chunk_start_byte..current_chunk_end_byte]
                        .to_string(),
                });
                current_chunk_start_byte = segment_range.start;
                current_chunk_end_byte = segment_range.end;
            } else {
                // Otherwise, extend the current chunk
                if current_chunk_start_byte == current_chunk_end_byte {
                    // First segment of a new chunk
                    current_chunk_start_byte = segment_range.start;
                }
                current_chunk_end_byte = segment_range.end;
            }
        }

        // Add the last chunk if it's not empty
        if current_chunk_start_byte != current_chunk_end_byte {
            chunks.push(TextChunk {
                repo: self.repo.clone(),
                path: self.path.clone(),
                sequence: chunks.len(),
                file_content: file_content.to_string(),
                chunk_text: file_content[current_chunk_start_byte..current_chunk_end_byte]
                    .to_string(),
            });
        }

        chunks
    }

    // This function recursively splits the text into segments based on delimiters.
    // It returns a vector of byte ranges (slices of the original text) that represent
    // "atomic" segments that cannot be further split by the current delimiter without
    // exceeding the max_chunk_size, or are the delimiters themselves.
    #[allow(clippy::only_used_in_recursion)]
    fn split_recursively_into_segments(
        &self,
        text: &str,
        delimiters: &[Regex],
        delimiter_idx: usize,
        max_chunk_size: usize,
        current_offset: usize, // New parameter: starting byte offset of 'text' in 'self.file_content'
    ) -> Vec<Range<usize>> {
        let mut result_segments: Vec<Range<usize>> = Vec::new();

        if text.is_empty() {
            return result_segments;
        }

        // Base case 1: If the text is already small enough, add its range.
        if text.len() <= max_chunk_size {
            result_segments.push(current_offset..(current_offset + text.len()));
            return result_segments;
        }

        // Base case 2: If we've exhausted all specified delimiters,
        // and the text is still too large, split by characters.
        if delimiter_idx >= delimiters.len() {
            let mut local_start = 0;
            while local_start < text.len() {
                let local_end = (local_start + max_chunk_size).min(text.len());
                result_segments.push(current_offset + local_start..current_offset + local_end);
                local_start = local_end;
            }
            return result_segments;
        }

        let current_delimiter = &delimiters[delimiter_idx];
        let mut local_byte_start = 0;

        for mat in current_delimiter.find_iter(text) {
            // Process the text before the delimiter
            if mat.start() > local_byte_start {
                let sub_text = &text[local_byte_start..mat.start()];
                result_segments.extend(self.split_recursively_into_segments(
                    sub_text,
                    delimiters,
                    delimiter_idx + 1, // Try next delimiter for sub-segments
                    max_chunk_size,
                    current_offset + local_byte_start, // Pass updated offset
                ));
            }
            // Add the delimiter itself as a segment
            result_segments
                .push(current_offset + mat.range().start..current_offset + mat.range().end);
            local_byte_start = mat.end();
        }

        // Process any remaining text after the last delimiter, but only if a split occurred.
        if local_byte_start < text.len() {
            let sub_text = &text[local_byte_start..];
            result_segments.extend(self.split_recursively_into_segments(
                sub_text,
                delimiters,
                delimiter_idx + 1,
                max_chunk_size,
                current_offset + local_byte_start, // Pass updated offset
            ));
        }

        result_segments
    }
}

impl TextChunk {
    /// Builds a formatted string representation of the text chunk, suitable for
    /// use as a context passage in retrieval models.
    ///
    /// The output format includes the repository and file path, followed by
    /// the chunk's text content within a `focus: {}` block. This format is
    /// designed to provide a clear, self-contained snippet for the model.
    ///
    /// # Returns
    ///
    /// A formatted `String` representing the chunk as a passage.
    ///
    /// # Examples
    ///
    /// ```
    /// use janet_ai_context::text::{TextContextBuilder, TextChunk};
    ///
    /// let repo_name = "my_repo".to_string();
    /// let file_path = "src/utils.rs".to_string();
    /// let delimiter_patterns = vec![r"\n\n", r"\n", r" "];
    /// let max_chunk_length = 500;
    /// let builder = TextContextBuilder::new(repo_name, file_path, &delimiter_patterns, max_chunk_length);
    ///
    /// let file_content = r#"
    /// pub fn add(a: i32, b: i32) -> i32 {
    ///     a + b
    /// }
    ///
    /// pub fn subtract(a: i32, b: i32) -> i32 {
    ///     a - b
    /// }
    /// "#;
    ///
    /// let chunks = builder.get_chunks(file_content);
    ///
    /// // Assuming the first chunk contains "pub fn add..."
    /// let first_chunk = &chunks[0];
    /// let passage = first_chunk.build();
    ///
    /// assert!(passage.starts_with("passage: {\"repo\": \"my_repo\", \"path\": \"src/utils.rs\"}"));
    /// // Note that focus is followed by two newlines, because one is part of the context format, and the other is from the original content.
    /// assert!(passage.contains("focus: {}\n\npub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}"));
    /// assert!(!passage.contains("context:")); // No context chunks in this format
    /// ```
    pub fn build(&self) -> String {
        format!(
            "passage: {{\"repo\": \"{}\", \"path\": \"{}\"}}\n\nfocus: {{}}\n{}",
            self.repo, self.path, self.chunk_text
        )
    }

    /// Get the formatted summary (same as build) for backwards compatibility
    pub fn summary(&self) -> String {
        self.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_chunks_basic() {
        let repo = "test_repo".to_string();
        let path = "test_path/document.md".to_string();
        // Use the default Markdown delimiters
        let delimiter_patterns = DEFAULT_MARKDOWN_DELIMITERS;
        // Procedurally generate a long string by repeating a simple sentence 100 times
        let file_content = (0..100)
            .map(|_| "This is a test sentence. ")
            .collect::<String>();
        let max_chunk_length = 500;
        let optimal_chunk_count = file_content.len() / max_chunk_length + 1;

        let builder = TextContextBuilder::new(
            repo.clone(),
            path.clone(),
            delimiter_patterns,
            max_chunk_length,
        );
        let chunks = builder.get_chunks(&file_content);

        // The max_chunk_size is 500. The content is ~1000 chars.
        // It should split into at least two chunks.
        assert!(!chunks.is_empty());
        assert!(chunks.len() >= optimal_chunk_count - 1);
        // It should not split into more than 3 chunks
        println!("Number of chunks: {}", chunks.len());
        assert!(chunks.len() <= optimal_chunk_count + 1);

        // Verify properties of the first chunk
        assert_eq!(chunks[0].repo, repo);
        assert_eq!(chunks[0].path, path);
        assert_eq!(chunks[0].file_content, file_content);
        assert!(chunks[0].chunk_text.len() <= max_chunk_length);

        // Verify properties of the last chunk
        let last_chunk_idx = chunks.len() - 1;
        assert_eq!(chunks[last_chunk_idx].repo, repo);
        assert_eq!(chunks[last_chunk_idx].path, path);
        assert_eq!(chunks[last_chunk_idx].file_content, file_content);
        assert!(chunks[last_chunk_idx].chunk_text.len() <= max_chunk_length);

        // Verify that concatenating chunks reconstructs the original content
        let reconstructed_content: String = chunks.iter().map(|c| c.chunk_text.as_str()).collect();
        assert_eq!(reconstructed_content, file_content);
    }

    #[test]
    fn test_get_chunks_single_chunk() {
        let repo = "test_repo".to_string();
        let path = "test_path/short_doc.md".to_string();
        let file_content = "This is a very short Markdown document."; // Less than max_chunk_size
        let delimiter_patterns = DEFAULT_MARKDOWN_DELIMITERS;
        let max_chunk_length = 500;

        let builder = TextContextBuilder::new(
            repo.clone(),
            path.clone(),
            delimiter_patterns,
            max_chunk_length,
        );
        let chunks = builder.get_chunks(file_content);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].repo, repo);
        assert_eq!(chunks[0].path, path);
        assert_eq!(chunks[0].file_content, file_content);
        assert_eq!(chunks[0].chunk_text, file_content);
    }

    #[test]
    fn test_get_chunks_empty_content() {
        let repo = "test_repo".to_string();
        let path = "test_path/empty_doc.md".to_string();
        let file_content = "";
        let delimiter_patterns = DEFAULT_MARKDOWN_DELIMITERS;
        let max_chunk_length = 500;

        let builder = TextContextBuilder::new(repo, path, delimiter_patterns, max_chunk_length);
        let chunks = builder.get_chunks(file_content);

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_get_chunks_with_markdown_elements() {
        let repo = "test_repo".to_string();
        let path = "test_path/markdown_doc.md".to_string();
        let file_content = r#"
# Heading 1

This is a paragraph.

## Heading 2

- List item 1
- List item 2

```rust
fn main() {
    println!("Hello, world!");
}
```

Another paragraph.
"#;
        let delimiter_patterns = DEFAULT_MARKDOWN_DELIMITERS;
        let max_chunk_length = 500;

        let builder = TextContextBuilder::new(
            repo.clone(),
            path.clone(),
            delimiter_patterns,
            max_chunk_length,
        );
        let chunks = builder.get_chunks(file_content);

        assert!(!chunks.is_empty());
        let reconstructed_content: String = chunks.iter().map(|c| c.chunk_text.as_str()).collect();
        assert_eq!(reconstructed_content, file_content);

        for chunk in &chunks {
            assert!(chunk.chunk_text.len() <= 500);
            assert_eq!(chunk.repo, repo);
            assert_eq!(chunk.path, path);
            assert_eq!(chunk.file_content, file_content);
        }

        // Verify that specific markdown elements are likely to be in their own chunks or at chunk boundaries
        let headings_chunk = chunks
            .iter()
            .find(|c| c.chunk_text.contains("# Heading 1"))
            .unwrap();
        assert!(headings_chunk.chunk_text.contains("This is a paragraph."));

        let list_chunk = chunks
            .iter()
            .find(|c| c.chunk_text.contains("- List item 1"))
            .unwrap();
        assert!(list_chunk.chunk_text.contains("- List item 2"));

        let code_block_chunk = chunks
            .iter()
            .find(|c| c.chunk_text.contains("```rust"))
            .unwrap();
        assert!(
            code_block_chunk
                .chunk_text
                .contains("println!(\"Hello, world!\");")
        );
    }
}
