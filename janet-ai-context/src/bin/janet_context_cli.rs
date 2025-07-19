use clap::Parser;
use janet_ai_context::text::{DEFAULT_MARKDOWN_DELIMITERS, TextChunk, TextContextBuilder};
use serde::Serialize;
use std::fs;
use std::io::{self, Read};

/// A CLI tool to chunk text files into JSON output using janet-ai-context.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input text file. If not provided, reads from stdin.
    #[arg(short, long)]
    input: Option<String>,

    /// Repository name for the context.
    #[arg(short, long, default_value = "unknown_repo")]
    repo: String,

    /// File path within the repository for the context.
    #[arg(short, long, default_value = "unknown_path")]
    path: String,

    /// Maximum length for each text chunk.
    #[arg(short, long, default_value_t = 5000)]
    max_chunk_length: usize,

    /// Comma-separated list of regex patterns for delimiters.
    /// Defaults to Markdown delimiters if not provided.
    #[arg(short, long, value_delimiter = ',')]
    delimiters: Option<Vec<String>>,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let file_content = if let Some(input_path) = args.input {
        fs::read_to_string(input_path)?
    } else {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer
    };

    let delimiter_patterns_owned: Vec<String> = if let Some(d) = args.delimiters {
        d
    } else {
        DEFAULT_MARKDOWN_DELIMITERS
            .iter()
            .map(|&s| s.to_string())
            .collect()
    };

    let delimiter_patterns_refs: Vec<&str> = delimiter_patterns_owned
        .iter()
        .map(|s| s.as_str())
        .collect();

    let builder = TextContextBuilder::new(
        args.repo,
        args.path,
        &delimiter_patterns_refs,
        args.max_chunk_length,
    );

    let chunks = builder.get_chunks(&file_content);

    #[derive(Serialize)]
    struct SerializableTextChunk<'a> {
        repo: &'a str,
        path: &'a str,
        sequence: usize,
        chunk_text: &'a str,
        summary: String,
    }

    let serializable_chunks: Vec<SerializableTextChunk> = chunks
        .into_iter()
        .map(|c| SerializableTextChunk {
            repo: c.repo,
            path: c.path,
            sequence: c.sequence,
            chunk_text: c.chunk_text,
            summary: c.build(),
        })
        .collect();

    let json_output = serde_json::to_string_pretty(&serializable_chunks)?;
    println!("{}", json_output);

    Ok(())
}
