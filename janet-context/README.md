# Janet Context

`janet-context` is a Rust library designed to facilitate the creation of structured text contexts for retrieval models, particularly within Retrieval Augmented Generation (RAG) systems. Its primary goal is to transform raw code or text content into manageable "passages" or "chunks" that can be effectively used for retrieval tasks, such as code search or documentation lookup.

The library focuses on segmenting content while preserving crucial metadata and ensuring chunks are optimized for model input limits.

## Core Concepts

### `TextContextBuilder`
This struct is the main entry point for configuring how text is chunked. It allows you to specify:
-   **Repository Name**: The name of the code repository.
-   **File Path**: The path to the file within the repository.
-   **Delimiters**: A set of regular expression patterns used to split the text into logical segments. These are applied recursively, prioritizing larger delimiters (e.g., double newlines, headings) before smaller ones (e.g., single newlines, spaces) to create meaningful chunks.
-   **Maximum Chunk Length**: The maximum character length for each generated text chunk.

### `TextChunk`
Represents a single segment of text extracted from a file. Each `TextChunk` includes:
-   `repo`: The repository name.
-   `path`: The file path.
-   `sequence`: The 0-indexed order of the chunk within the file.
-   `file_content`: A reference to the entire original file content.
-   `chunk_text`: The actual text content of this specific chunk.

`TextChunk` also provides a `build` method to format the chunk into a string suitable for retrieval models, typically including metadata like `passage: {"repo": "...", "path": "..."}` and the chunk's content within a `focus: {}` block.

## Example Context Format

The library aims to produce code snippets that look like this for retrieval models:

```rs
passage: {"repo": "example", "path": "the_crate/src/lib.rs"}

context: {"module": "self"}
/// Create a wrapped database connection
use std::sync::{Arc, Mutex};
use anyhow::Result;
use some_crate::module::DBConnection;
use other_crate::stuff;
use crate::sync::DatabaseHandle;
// ... (more code)

context: {"module": "crate::sync"}
struct DatabaseHandle {
    thing: Arc<Mutex<DBConnection>>,
    // ... (more fields)
}

focus: {}
pub fn connect(host: str) -> Result<DatabaseHandle> {
    DatabaseHandle {
        thing: Arc::new(Mutex::new(DBConnection::connect(host)?))
    }
}
```

## Key Features

*   **Metadata Inclusion**: Each passage includes repository and file path information.
*   **Contextual Segmentation**: Text is split into segments based on configurable regex delimiters to create meaningful chunks.
*   **Recursive Splitting**: Intelligently breaks down text, prioritizing larger delimiters first, ensuring chunks do not exceed a specified maximum size.
*   **Chunk Reconstruction**: Guarantees that all original file content can be reconstructed by concatenating the generated `TextChunk`s.
*   **Default Markdown Delimiters**: Provides a set of default regex patterns optimized for splitting Markdown content.

## Usage (Library)

```rust
use janet_context::text::{TextContextBuilder, TextChunk};

let repo_name = "my_rust_project".to_string();
let file_path = "src/main.rs".to_string();
// Delimiters are used to split the text into logical segments.
// They are applied in order: try splitting by double newline, then single newline, then space.
let delimiter_patterns = vec![r"\n\n", r"\n", r" "];
let max_chunk_length = 500;

let builder = TextContextBuilder::new(repo_name, file_path, &delimiter_patterns, max_chunk_length);

let file_content = r#"
fn main() {
    println!("Hello, world!");

    // This is another part of the code.
    let x = 10;
    let y = 20;
    println!("Sum: {}", x + y);
}
"#;

let chunks = builder.get_chunks(file_content);

// Verify that chunks were created and their content is correct
assert!(!chunks.is_empty());
let reconstructed_content: String = chunks.iter().map(|c| c.chunk_text).collect();
assert_eq!(reconstructed_content, file_content);

// You can iterate over chunks to process them individually
for chunk in chunks {
    println!("Chunk sequence: {}", chunk.sequence);
    println!("Chunk text: \"{}\"", &chunk.chunk_text);
    // Further processing of each chunk can happen here
}
```

## Command Line Interface (CLI)

The `janet-context` crate also provides a command-line tool (`janet_context_cli`) for chunking text files into JSON output. This is useful for quickly processing files and integrating with other tools.

### Installation

To install the CLI tool, ensure you have Rust and Cargo installed, then run:

```bash
cargo install --path janet-context
```

### Usage (CLI)

```bash
janet_context_cli [OPTIONS]
```

#### Arguments:

*   `-i, --input <INPUT>`: Path to the input text file. If not provided, the tool reads from standard input (stdin).
*   `-r, --repo <REPO>`: Repository name for the context. Defaults to `"unknown_repo"`.
*   `-p, --path <PATH>`: File path within the repository for the context. Defaults to `"unknown_path"`.
*   `-m, --max-chunk-length <MAX_CHUNK_LENGTH>`: Maximum length for each text chunk. Defaults to `500`.
*   `-d, --delimiters <DELIMITERS>`: Comma-separated list of regex patterns for delimiters. If not provided, it defaults to the library's `DEFAULT_MARKDOWN_DELIMITERS`.

#### Examples:

1.  **Chunking a file with default settings:**
    ```bash
    janet_context_cli --input my_code.rs
    ```

2.  **Chunking from stdin with custom repo and path:**
    ```bash
    cat my_document.md | janet_context_cli --repo "my_docs" --path "docs/intro.md"
    ```

3.  **Specifying a custom maximum chunk length:**
    ```bash
    janet_context_cli --input large_file.txt --max-chunk-length 1000
    ```

4.  **Using custom delimiters (e.g., splitting only by newlines and spaces):**
    ```bash
    janet_context_cli --input my_text.txt --delimiters "\n, "
    ```

The CLI outputs a JSON array of `TextChunk` objects to standard output, each containing `repo`, `path`, `sequence`, and `chunk_text`.
