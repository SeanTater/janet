use anyhow::Result;
use janet_ai_context::text::{TextContextBuilder, DEFAULT_MARKDOWN_DELIMITERS};
use std::path::Path;

/// Information about a chunk that owns its data (no lifetimes)
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// The name of the repository the file belongs to
    pub repo: String,
    /// The path to the file within the repository
    pub path: String,
    /// The sequence number of this chunk within the file (0-indexed)
    pub sequence: usize,
    /// The text content of this specific chunk
    pub chunk_text: String,
}

/// Configuration for different chunking strategies based on file types
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Maximum size of each chunk in characters
    pub max_chunk_size: usize,
    /// Repository name for context
    pub repo_name: String,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 2000, // Reasonable size for code chunks
            repo_name: "unknown".to_string(),
        }
    }
}

impl ChunkingConfig {
    pub fn new(repo_name: String) -> Self {
        Self {
            max_chunk_size: 2000,
            repo_name,
        }
    }
    
    pub fn with_max_chunk_size(mut self, max_chunk_size: usize) -> Self {
        self.max_chunk_size = max_chunk_size;
        self
    }
}

/// Represents different types of content that require different chunking strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentType {
    /// Rust source code
    Rust,
    /// JavaScript/TypeScript code
    JavaScript,
    /// Python code
    Python,
    /// Go code
    Go,
    /// C/C++ code
    C,
    /// Java code
    Java,
    /// Markdown documentation
    Markdown,
    /// Plain text files
    Text,
    /// JSON data files
    Json,
    /// YAML/TOML configuration files
    Config,
    /// Unknown file type (treated as text)
    Unknown,
}

impl ContentType {
    /// Determine content type from file path
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("rs") => Self::Rust,
            Some("js") | Some("jsx") | Some("ts") | Some("tsx") | Some("mjs") => Self::JavaScript,
            Some("py") | Some("pyi") => Self::Python,
            Some("go") => Self::Go,
            Some("c") | Some("cpp") | Some("cc") | Some("cxx") | Some("h") | Some("hpp") => Self::C,
            Some("java") => Self::Java,
            Some("md") | Some("markdown") => Self::Markdown,
            Some("txt") | Some("text") => Self::Text,
            Some("json") => Self::Json,
            Some("yaml") | Some("yml") | Some("toml") => Self::Config,
            _ => {
                // Check filename patterns
                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                    if filename.starts_with("README") || filename.starts_with("CHANGELOG") {
                        return Self::Markdown;
                    }
                    if filename == "Cargo.toml" || filename == "pyproject.toml" {
                        return Self::Config;
                    }
                    if filename == "package.json" || filename.ends_with(".json") {
                        return Self::Json;
                    }
                }
                Self::Unknown
            }
        }
    }
    
    /// Get appropriate delimiters for this content type
    pub fn get_delimiters(&self) -> &'static [&'static str] {
        match self {
            Self::Rust => &[
                r"(?m)^pub\s+(struct|enum|trait|fn)", // Public declarations
                r"(?m)^impl\b",                        // Implementation blocks
                r"(?m)^fn\s+\w+",                      // Function definitions
                r"(?m)^struct\s+\w+",                  // Struct definitions
                r"(?m)^enum\s+\w+",                    // Enum definitions
                r"(?m)^trait\s+\w+",                   // Trait definitions
                r"(?m)^mod\s+\w+",                     // Module definitions
                r"(?m)^use\s+",                        // Use statements
                r"\n\n",                               // Paragraph breaks
                r"\n",                                 // Line breaks
                r" ",                                  // Word boundaries
            ],
            Self::JavaScript => &[
                r"(?m)^export\s+(class|function|const|let|var)", // Exports
                r"(?m)^class\s+\w+",                              // Class definitions
                r"(?m)^function\s+\w+",                           // Function definitions
                r"(?m)^const\s+\w+\s*=\s*(?:\([^)]*\)\s*=>|\([^)]*\)\s*{)", // Arrow functions
                r"(?m)^import\s+",                                // Import statements
                r"\n\n",                                          // Paragraph breaks
                r"\n",                                            // Line breaks
                r" ",                                             // Word boundaries
            ],
            Self::Python => &[
                r"(?m)^class\s+\w+",                    // Class definitions
                r"(?m)^def\s+\w+",                      // Function definitions
                r"(?m)^async\s+def\s+\w+",              // Async function definitions
                r"(?m)^from\s+\w+\s+import",            // From imports
                r"(?m)^import\s+\w+",                   // Import statements
                r#"(?m)^if\s+__name__\s*==\s*['"]__main__['"]:"#, // Main blocks
                r"\n\n",                                // Paragraph breaks
                r"\n",                                  // Line breaks
                r" ",                                   // Word boundaries
            ],
            Self::Go => &[
                r"(?m)^package\s+\w+",           // Package declarations
                r"(?m)^func\s+\w+",              // Function definitions
                r"(?m)^func\s+\([^)]*\)\s+\w+", // Method definitions
                r"(?m)^type\s+\w+\s+struct",     // Struct definitions
                r"(?m)^type\s+\w+\s+interface",  // Interface definitions
                r"(?m)^import\s*\(",             // Import blocks
                r"(?m)^var\s+\w+",               // Variable declarations
                r"(?m)^const\s+\w+",             // Constant declarations
                r"\n\n",                         // Paragraph breaks
                r"\n",                           // Line breaks
                r" ",                            // Word boundaries
            ],
            Self::Java => &[
                r"(?m)^package\s+",               // Package declarations
                r"(?m)^import\s+",                // Import statements
                r"(?m)^public\s+class\s+\w+",     // Public class definitions
                r"(?m)^class\s+\w+",              // Class definitions
                r"(?m)^interface\s+\w+",          // Interface definitions
                r"(?m)^public\s+[^{]*\s+\w+\s*\(", // Public method definitions
                r"(?m)^private\s+[^{]*\s+\w+\s*\(", // Private method definitions
                r"\n\n",                          // Paragraph breaks
                r"\n",                            // Line breaks
                r" ",                             // Word boundaries
            ],
            Self::C => &[
                r"(?m)^#include\s*<",    // System includes
                r#"(?m)^#include\s*""#,   // Local includes
                r"(?m)^#define\s+\w+",   // Macro definitions
                r"(?m)^typedef\s+",      // Type definitions
                r"(?m)^struct\s+\w+",    // Struct definitions
                r"(?m)^enum\s+\w+",      // Enum definitions
                r"(?m)^\w+\s+\w+\s*\(", // Function definitions
                r"\n\n",                 // Paragraph breaks
                r"\n",                   // Line breaks
                r" ",                    // Word boundaries
            ],
            Self::Markdown => DEFAULT_MARKDOWN_DELIMITERS,
            Self::Json => &[
                r",\s*\n",  // JSON object/array separators
                r"\n",      // Line breaks
                r" ",       // Spaces
            ],
            Self::Config => &[
                r"(?m)^\[.*\]",  // TOML/INI sections
                r"(?m)^---",     // YAML document separators
                r"\n\n",         // Paragraph breaks
                r"\n",           // Line breaks
                r" ",            // Spaces
            ],
            Self::Text | Self::Unknown => DEFAULT_MARKDOWN_DELIMITERS,
        }
    }
    
    /// Get a reasonable chunk size for this content type
    pub fn get_default_chunk_size(&self) -> usize {
        match self {
            Self::Rust | Self::JavaScript | Self::Python | Self::Go | Self::Java | Self::C => 1500, // Code needs smaller chunks
            Self::Markdown | Self::Text => 3000, // Documentation can have larger chunks
            Self::Json | Self::Config => 1000,   // Data files should be split conservatively
            Self::Unknown => 2000,               // Default middle ground
        }
    }
    
    /// Check if this content type should be indexed
    pub fn should_index(&self) -> bool {
        match self {
            Self::Unknown => false, // Don't index unknown file types by default
            _ => true,
        }
    }
}

/// Strategy for chunking files based on their content type
#[derive(Clone)]
pub struct ChunkingStrategy {
    config: ChunkingConfig,
}

impl ChunkingStrategy {
    /// Create a new chunking strategy with the given configuration
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }
    
    /// Chunk a file's content based on its path and content type
    pub fn chunk_content(
        &self,
        file_path: &Path,
        content: &str,
    ) -> Result<Vec<ChunkInfo>> {
        let content_type = ContentType::from_path(file_path);
        
        // Skip indexing for unsupported content types
        if !content_type.should_index() {
            return Ok(vec![]);
        }
        
        let delimiters = content_type.get_delimiters();
        let chunk_size = std::cmp::min(
            self.config.max_chunk_size,
            content_type.get_default_chunk_size(),
        );
        
        let builder = TextContextBuilder::new(
            self.config.repo_name.clone(),
            file_path.to_string_lossy().to_string(),
            delimiters,
            chunk_size,
        );
        
        let chunks = builder.get_chunks(content);
        
        // Convert TextChunk to ChunkInfo (owned data)
        let chunk_infos: Vec<ChunkInfo> = chunks
            .into_iter()
            .map(|chunk| ChunkInfo {
                repo: chunk.repo.to_string(),
                path: chunk.path.to_string(),
                sequence: chunk.sequence,
                chunk_text: chunk.chunk_text.to_string(),
            })
            .collect();
        
        tracing::debug!(
            "Chunked {} ({:?}) into {} chunks (max size: {})",
            file_path.display(),
            content_type,
            chunk_infos.len(),
            chunk_size
        );
        
        Ok(chunk_infos)
    }
    
    /// Get the content type for a given file path
    pub fn get_content_type(&self, file_path: &Path) -> ContentType {
        ContentType::from_path(file_path)
    }
    
    /// Check if a file should be indexed based on its path
    pub fn should_index_file(&self, file_path: &Path) -> bool {
        let content_type = ContentType::from_path(file_path);
        content_type.should_index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_content_type_detection() {
        assert_eq!(ContentType::from_path(Path::new("src/lib.rs")), ContentType::Rust);
        assert_eq!(ContentType::from_path(Path::new("app.js")), ContentType::JavaScript);
        assert_eq!(ContentType::from_path(Path::new("main.py")), ContentType::Python);
        assert_eq!(ContentType::from_path(Path::new("main.go")), ContentType::Go);
        assert_eq!(ContentType::from_path(Path::new("App.java")), ContentType::Java);
        assert_eq!(ContentType::from_path(Path::new("main.c")), ContentType::C);
        assert_eq!(ContentType::from_path(Path::new("README.md")), ContentType::Markdown);
        assert_eq!(ContentType::from_path(Path::new("config.json")), ContentType::Json);
        assert_eq!(ContentType::from_path(Path::new("Cargo.toml")), ContentType::Config);
        assert_eq!(ContentType::from_path(Path::new("unknown.xyz")), ContentType::Unknown);
        
        // Test special filenames
        assert_eq!(ContentType::from_path(Path::new("README")), ContentType::Markdown);
        assert_eq!(ContentType::from_path(Path::new("CHANGELOG")), ContentType::Markdown);
        assert_eq!(ContentType::from_path(Path::new("package.json")), ContentType::Json);
    }
    
    #[test]
    fn test_content_type_properties() {
        assert!(ContentType::Rust.should_index());
        assert!(ContentType::Markdown.should_index());
        assert!(!ContentType::Unknown.should_index());
        
        assert_eq!(ContentType::Rust.get_default_chunk_size(), 1500);
        assert_eq!(ContentType::Markdown.get_default_chunk_size(), 3000);
        assert_eq!(ContentType::Json.get_default_chunk_size(), 1000);
    }
    
    #[test]
    fn test_chunking_strategy() {
        let config = ChunkingConfig::new("test_repo".to_string());
        let strategy = ChunkingStrategy::new(config);
        
        let rust_content = r#"
use std::collections::HashMap;

pub struct MyStruct {
    field: String,
}

impl MyStruct {
    pub fn new(field: String) -> Self {
        Self { field }
    }
    
    pub fn get_field(&self) -> &str {
        &self.field
    }
}

fn main() {
    let s = MyStruct::new("test".to_string());
    println!("{}", s.get_field());
}
"#;
        
        let rust_path = PathBuf::from("src/main.rs");
        let chunks = strategy.chunk_content(&rust_path, rust_content).unwrap();
        
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].repo, "test_repo");
        assert_eq!(chunks[0].path, "src/main.rs");
        
        // Verify that all chunks combined reconstruct the original content
        let reconstructed: String = chunks.iter().map(|c| c.chunk_text.clone()).collect();
        assert_eq!(reconstructed, rust_content);
    }
    
    #[test]
    fn test_should_index_file() {
        let config = ChunkingConfig::new("test".to_string());
        let strategy = ChunkingStrategy::new(config);
        
        assert!(strategy.should_index_file(Path::new("src/lib.rs")));
        assert!(strategy.should_index_file(Path::new("README.md")));
        assert!(strategy.should_index_file(Path::new("config.json")));
        assert!(!strategy.should_index_file(Path::new("binary.exe")));
        assert!(!strategy.should_index_file(Path::new("unknown.xyz")));
    }
}