//! Working demonstration of the indexing system without embedding dependencies
//! 
//! This example shows the core functionality:
//! 1. Setting up an IndexingEngine 
//! 2. Creating test files with various content
//! 3. Indexing the files and chunking text
//! 4. Verifying the indexing worked correctly
//! 5. Demonstrating basic search capability (text-based)

use anyhow::Result;
use janet_ai_retriever::retrieval::{
    indexing_engine::{IndexingEngine, IndexingEngineConfig},
    indexing_mode::IndexingMode,
};
use sqlx::Row;
use std::path::PathBuf;
use tempfile::tempdir;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for better visibility
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Starting working indexing demo...\n");

    // Create a temporary directory for our test repository
    let temp_dir = tempdir()?;
    let repo_path = temp_dir.path().to_path_buf();
    
    println!("📁 Created test repository at: {}", repo_path.display());

    // Create realistic test files
    create_demo_files(&repo_path).await?;
    
    println!("📝 Created demo files with various programming content\n");

    // Set up the indexing engine (no embeddings for this demo)
    let indexing_config = IndexingEngineConfig::new(
        "demo-repo".to_string(),
        repo_path.clone(),
    )
    .with_mode(IndexingMode::FullReindex)
    .with_max_workers(2)
    .with_chunk_size(300); // Smaller chunks for demonstration

    println!("⚙️  Initializing IndexingEngine...");
    
    // Create indexing engine with in-memory database
    let mut engine = IndexingEngine::new_memory(indexing_config).await?;
    
    println!("✅ IndexingEngine initialized successfully");

    // Start the engine and perform full reindex
    println!("🔄 Starting full reindex...");
    engine.start().await?;

    // Wait for indexing to complete
    let mut attempts = 0;
    let max_attempts = 30;
    
    loop {
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        // Process any pending tasks
        engine.process_pending_tasks().await?;
        
        let queue_size = engine.get_queue_size().await;
        let stats = engine.get_stats().await;
        
        if attempts % 5 == 0 { // Print status every second
            println!("📊 Queue: {}, Files: {}, Chunks: {}, Errors: {}", 
                    queue_size, stats.files_processed, stats.chunks_created, stats.errors);
        }
        
        if queue_size == 0 && stats.files_processed > 0 {
            println!("✅ Indexing completed!");
            break;
        }
        
        attempts += 1;
        if attempts >= max_attempts {
            println!("⚠️  Timeout waiting for indexing to complete");
            break;
        }
    }

    // Get final statistics
    let final_stats = engine.get_stats().await;
    let index_stats = engine.get_index_stats().await?;
    
    println!("\n📈 Final Statistics:");
    println!("   Files processed: {}", final_stats.files_processed);
    println!("   Chunks created: {}", final_stats.chunks_created);
    println!("   Processing errors: {}", final_stats.errors);
    println!("   Total files in index: {}", index_stats.files_count);
    println!("   Total chunks in index: {}", index_stats.chunks_count);

    // Demonstrate chunk retrieval
    println!("\n🔍 Demonstrating chunk retrieval...");
    
    let enhanced_index = engine.get_enhanced_index();
    
    // Get all chunks to show what was indexed
    let rows: Vec<sqlx::sqlite::SqliteRow> = sqlx::query("SELECT relative_path, line_start, line_end, content FROM chunks ORDER BY relative_path, line_start")
        .fetch_all(enhanced_index.file_index().pool())
        .await?;
    
    if rows.is_empty() {
        println!("❌ No chunks found in the index!");
    } else {
        println!("📄 Found {} chunks in the index:", rows.len());
        
        for (i, row) in rows.iter().enumerate() {
            let relative_path: String = row.get("relative_path");
            let line_start: i64 = row.get("line_start");
            let line_end: i64 = row.get("line_end");
            let content: String = row.get("content");
            
            let preview = if content.len() > 80 {
                format!("{}...", &content[..77])
            } else {
                content.clone()
            };
            
            println!("   {}. {}:{}-{} | {}", 
                    i + 1, 
                    relative_path,
                    line_start,
                    line_end,
                    preview.replace('\n', " "));
            
            if i >= 9 { // Show only first 10 chunks
                println!("   ... and {} more chunks", rows.len() - 10);
                break;
            }
        }
    }

    // Demonstrate text-based searching (without embeddings)
    println!("\n🔎 Demonstrating text-based search...");
    
    let search_terms = vec!["function", "class", "import", "async", "test"];
    
    for term in search_terms {
        let query = format!("SELECT relative_path, line_start, content FROM chunks WHERE content LIKE '%{}%' LIMIT 3", term);
        let search_results = sqlx::query(&query)
            .fetch_all(enhanced_index.file_index().pool())
            .await?;
        
        println!("\n   🔍 Search for '{}' found {} results:", term, search_results.len());
        
        for result in search_results {
            let relative_path: String = result.get("relative_path");
            let line_start: i64 = result.get("line_start");
            let content: String = result.get("content");
            
            let preview = if content.len() > 60 {
                format!("{}...", &content[..57])
            } else {
                content
            };
            
            println!("      📝 {}:{} | {}", 
                    relative_path, 
                    line_start,
                    preview.replace('\n', " "));
        }
    }

    // Clean up
    engine.shutdown().await?;
    
    println!("\n🎉 Demo completed successfully!");
    println!("   The indexing system demonstrated:");
    println!("   ✓ File discovery and processing");
    println!("   ✓ Text chunking with janet-ai-context");
    println!("   ✓ Storage in SQLite database");
    println!("   ✓ Basic text-based search capabilities");
    println!("   ✓ Priority-based task queue system");
    
    if final_stats.files_processed > 0 && final_stats.chunks_created > 0 {
        println!("\n🌟 The system is working correctly and ready for embedding integration!");
    } else {
        println!("\n⚠️  Some issues detected - check the logs above");
    }

    Ok(())
}

/// Create demo files with realistic programming content
async fn create_demo_files(repo_path: &PathBuf) -> Result<()> {
    // Create Rust math utilities
    tokio::fs::write(
        repo_path.join("math.rs"),
        r#"//! Mathematical utility functions

use std::f64::consts::PI;

/// Calculate the area of a circle
pub fn circle_area(radius: f64) -> f64 {
    PI * radius * radius
}

/// Calculate factorial using recursion
pub fn factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n * factorial(n - 1),
    }
}

/// Find the greatest common divisor
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_area() {
        assert!((circle_area(1.0) - PI).abs() < 1e-10);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(5), 120);
    }
}
"#,
    ).await?;

    // Create Python data processing
    tokio::fs::write(
        repo_path.join("data_processor.py"),
        r#"""Data processing utilities for analytics."""

import json
import asyncio
from typing import List, Dict, Any, Optional

class DataProcessor:
    """Process and analyze data from various sources."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.processed_count = 0
    
    async def process_async(self, data: List[Dict]) -> List[Dict]:
        """Process data asynchronously for better performance."""
        tasks = [self._process_item(item) for item in data]
        results = await asyncio.gather(*tasks)
        self.processed_count += len(results)
        return results
    
    async def _process_item(self, item: Dict) -> Dict:
        """Process a single data item."""
        # Simulate async processing
        await asyncio.sleep(0.001)
        
        processed = item.copy()
        processed['processed'] = True
        processed['timestamp'] = self._get_timestamp()
        return processed
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def analyze_trends(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in the processed data."""
        if not data:
            return {}
        
        # Calculate basic statistics
        numeric_fields = [k for k, v in data[0].items() if isinstance(v, (int, float))]
        analysis = {}
        
        for field in numeric_fields:
            values = [item[field] for item in data if field in item]
            if values:
                analysis[field] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return analysis

def load_data_from_file(filepath: str) -> List[Dict]:
    """Load data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {filepath}: {e}")
        return []

# Example usage
if __name__ == "__main__":
    processor = DataProcessor({'debug': True})
    sample_data = [
        {'id': 1, 'value': 10, 'category': 'A'},
        {'id': 2, 'value': 20, 'category': 'B'},
        {'id': 3, 'value': 30, 'category': 'A'},
    ]
    
    analysis = processor.analyze_trends(sample_data)
    print("Analysis results:", analysis)
"#,
    ).await?;

    // Create JavaScript API client
    tokio::fs::write(
        repo_path.join("api_client.js"),
        r#"/**
 * HTTP API client with authentication and error handling
 */

class ApiClient {
    constructor(baseUrl, options = {}) {
        this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
        this.headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };
        this.timeout = options.timeout || 10000;
        this.retryAttempts = options.retryAttempts || 3;
    }

    async request(method, endpoint, data = null) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const config = {
            method,
            headers: this.headers,
            timeout: this.timeout
        };

        if (data && ['POST', 'PUT', 'PATCH'].includes(method)) {
            config.body = JSON.stringify(data);
        }

        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            try {
                const response = await fetch(url, config);
                
                if (!response.ok) {
                    throw new ApiError(`HTTP ${response.status}: ${response.statusText}`, response.status);
                }

                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return await response.json();
                } else {
                    return await response.text();
                }
            } catch (error) {
                if (attempt === this.retryAttempts) {
                    throw error;
                }
                
                console.warn(`Request attempt ${attempt} failed:`, error.message);
                await this.delay(Math.pow(2, attempt) * 1000); // Exponential backoff
            }
        }
    }

    async get(endpoint) {
        return this.request('GET', endpoint);
    }

    async post(endpoint, data) {
        return this.request('POST', endpoint, data);
    }

    async put(endpoint, data) {
        return this.request('PUT', endpoint, data);
    }

    async delete(endpoint) {
        return this.request('DELETE', endpoint);
    }

    setAuthToken(token) {
        this.headers['Authorization'] = `Bearer ${token}`;
    }

    clearAuthToken() {
        delete this.headers['Authorization'];
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

class ApiError extends Error {
    constructor(message, status = null) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
    }
}

// Export for both Node.js and browsers
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ApiClient, ApiError };
} else {
    window.ApiClient = ApiClient;
    window.ApiError = ApiError;
}
"#,
    ).await?;

    // Create TypeScript interfaces
    tokio::fs::write(
        repo_path.join("types.ts"),
        r#"// Type definitions for the application

export interface User {
    id: number;
    username: string;
    email: string;
    createdAt: Date;
    lastLogin?: Date;
    roles: Role[];
}

export interface Role {
    id: number;
    name: string;
    permissions: Permission[];
}

export interface Permission {
    id: number;
    name: string;
    resource: string;
    action: 'create' | 'read' | 'update' | 'delete';
}

export interface ApiResponse<T> {
    success: boolean;
    data: T;
    message?: string;
    errors?: string[];
    meta?: {
        total: number;
        page: number;
        limit: number;
    };
}

export interface SearchQuery {
    query: string;
    filters?: Record<string, any>;
    sort?: {
        field: string;
        direction: 'asc' | 'desc';
    };
    pagination?: {
        page: number;
        limit: number;
    };
}

export interface SearchResult<T> {
    items: T[];
    total: number;
    hasMore: boolean;
    searchTime: number;
}

export class UserService {
    private apiClient: any;

    constructor(apiClient: any) {
        this.apiClient = apiClient;
    }

    async getUser(id: number): Promise<User> {
        const response = await this.apiClient.get(`/users/${id}`);
        return response.data;
    }

    async createUser(userData: Partial<User>): Promise<User> {
        const response = await this.apiClient.post('/users', userData);
        return response.data;
    }

    async updateUser(id: number, userData: Partial<User>): Promise<User> {
        const response = await this.apiClient.put(`/users/${id}`, userData);
        return response.data;
    }

    async deleteUser(id: number): Promise<void> {
        await this.apiClient.delete(`/users/${id}`);
    }

    async searchUsers(query: SearchQuery): Promise<SearchResult<User>> {
        const response = await this.apiClient.post('/users/search', query);
        return response.data;
    }
}
"#,
    ).await?;

    // Create a README
    tokio::fs::write(
        repo_path.join("README.md"),
        r#"# Demo Repository

This is a demonstration repository showing the capabilities of the janet-ai-retriever indexing system.

## Files Included

### Rust (`math.rs`)
Mathematical utility functions demonstrating:
- Function definitions with documentation
- Unit tests 
- Mathematical calculations
- Error handling patterns

### Python (`data_processor.py`)
Data processing utilities featuring:
- Async/await patterns
- Class definitions with methods
- Type hints and documentation
- JSON handling and file operations

### JavaScript (`api_client.js`)
HTTP API client showcasing:
- ES6 class syntax
- Promise-based async operations
- Error handling and retry logic
- Cross-platform compatibility

### TypeScript (`types.ts`)
Type definitions and service classes including:
- Interface definitions
- Generic types
- Class implementations
- Service layer patterns

## Indexing Features Demonstrated

The indexing system processes these files to:

1. **Text Chunking**: Breaks down large files into manageable chunks
2. **Language Detection**: Identifies different programming languages
3. **Structure Preservation**: Maintains code structure and context
4. **Metadata Extraction**: Captures file paths, line numbers, and content
5. **Search Capabilities**: Enables fast text-based searching

## Search Examples

You can search for various programming concepts:

- Functions: `function`, `def`, `fn`
- Classes: `class`, `interface`, `struct`
- Async patterns: `async`, `await`, `Promise`
- Error handling: `try`, `catch`, `Error`, `Result`
- Types: `string`, `number`, `boolean`, `Dict`, `List`

The system will find relevant code snippets across all indexed files.
"#,
    ).await?;

    Ok(())
}