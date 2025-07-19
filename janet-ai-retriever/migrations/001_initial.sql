-- Initial schema for janet-retriever database

-- Files table for tracking indexed files
CREATE TABLE IF NOT EXISTS files (
    hash BLOB PRIMARY KEY,
    relative_path TEXT UNIQUE NOT NULL,
    size INTEGER NOT NULL,
    modified_at TIMESTAMP NOT NULL,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table for storing code segments
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash BLOB NOT NULL,
    relative_path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB, -- Store embeddings as packed f32 array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_chunk UNIQUE(file_hash, line_start, line_end),
    FOREIGN KEY (file_hash) REFERENCES files(hash) ON DELETE CASCADE
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(relative_path);
CREATE INDEX IF NOT EXISTS idx_files_path ON files(relative_path);
CREATE INDEX IF NOT EXISTS idx_files_modified ON files(modified_at);

-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;