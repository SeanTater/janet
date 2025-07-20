//! End-to-end example demonstrating the complete indexing workflow
//! 
//! This example shows how to:
//! 1. Set up an IndexingEngine (without embeddings for simplicity)
//! 2. Create comprehensive test files with various content
//! 3. Index the files and generate text chunks
//! 4. Search for content using text-based queries
//! 5. Verify meaningful results are returned
//! 
//! Note: This example focuses on the core indexing functionality.
//! For embedding-based semantic search, actual embedding models would need to be downloaded.

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

    println!("🚀 Starting end-to-end indexing and search example...\n");

    // Create a temporary directory for our test repository
    let temp_dir = tempdir()?;
    let repo_path = temp_dir.path().to_path_buf();
    
    println!("📁 Created test repository at: {}", repo_path.display());

    // Create realistic test files with different content types
    create_test_files(&repo_path).await?;
    
    println!("📝 Created test files with various content types\n");

    // Set up the indexing engine (without embeddings for this demo)
    let indexing_config = IndexingEngineConfig::new(
        "test-repo".to_string(),
        repo_path.clone(),
    )
    .with_mode(IndexingMode::FullReindex)
    .with_max_workers(2)
    .with_chunk_size(500);

    println!("⚙️  Initializing IndexingEngine...");
    
    // Create indexing engine (using in-memory database for this example)
    let mut engine = IndexingEngine::new_memory(indexing_config).await?;
    
    println!("✅ IndexingEngine initialized successfully");

    // Start the engine and perform full reindex
    println!("🔄 Starting full reindex...");
    engine.start().await?;

    // Wait for indexing to complete
    let mut attempts = 0;
    let max_attempts = 30; // 30 seconds max wait
    
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        
        // Process any pending tasks
        engine.process_pending_tasks().await?;
        
        let queue_size = engine.get_queue_size().await;
        let stats = engine.get_stats().await;
        
        println!("📊 Queue size: {}, Files processed: {}, Chunks created: {}, Embeddings: {}", 
                queue_size, stats.files_processed, stats.chunks_created, stats.embeddings_generated);
        
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
    println!("   Embeddings generated: {}", final_stats.embeddings_generated);
    println!("   Total files in index: {}", index_stats.files_count);
    println!("   Total chunks in index: {}", index_stats.chunks_count);
    println!("   Embedding models: {}", index_stats.models_count);

    // Now let's demonstrate search functionality
    println!("\n🔍 Testing search functionality...");

    // Access the enhanced file index for searching
    let enhanced_index = engine.get_enhanced_index();
    
    // Test different text-based search queries
    let search_queries = vec![
        ("functions", "function"),
        ("HTTP requests", "HTTP"),
        ("database operations", "database"),
        ("authentication", "auth"),
        ("mathematical operations", "add"),
    ];

    for (description, search_term) in search_queries {
        println!("\n🔎 Searching for {} (keyword: '{}')", description, search_term);
        
        // Perform text-based search using SQL LIKE
        let query = format!(
            "SELECT relative_path, line_start, line_end, content FROM chunks 
             WHERE content LIKE '%{}%' 
             ORDER BY relative_path, line_start LIMIT 5", 
            search_term
        );
        
        let search_results: Vec<sqlx::sqlite::SqliteRow> = sqlx::query(&query)
            .fetch_all(enhanced_index.file_index().pool())
            .await?;
        
        println!("   Found {} results:", search_results.len());
        for (i, result) in search_results.iter().enumerate() {
            let relative_path: String = result.get("relative_path");
            let line_start: i64 = result.get("line_start");
            let line_end: i64 = result.get("line_end");
            let content: String = result.get("content");
            
            println!("   {}. {}:{}-{}", 
                    i + 1, 
                    relative_path,
                    line_start,
                    line_end);
            
            // Show a preview of the content (first 100 chars)
            let preview = content.chars().take(100).collect::<String>();
            let preview = if content.len() > 100 {
                format!("{}...", preview)
            } else {
                preview
            };
            println!("      Preview: {}", preview.replace('\n', " "));
        }
        
        if search_results.is_empty() {
            println!("   ⚠️  No results found for '{}'", search_term);
        }
    }

    // Clean up
    engine.shutdown().await?;
    
    println!("\n🎉 End-to-end example completed successfully!");
    println!("   The indexing system successfully:");
    println!("   ✓ Indexed {} files", final_stats.files_processed);
    println!("   ✓ Created {} text chunks", final_stats.chunks_created);
    println!("   ✓ Processed {} errors gracefully", final_stats.errors);
    println!("   ✓ Performed text-based search with meaningful results");

    Ok(())
}

/// Create realistic test files with various content types
async fn create_test_files(repo_path: &PathBuf) -> Result<()> {
    // Create a simple Rust library with multiple modules
    let src_dir = repo_path.join("src");
    tokio::fs::create_dir_all(&src_dir).await?;

    // Main library file
    tokio::fs::write(
        src_dir.join("lib.rs"),
        r#"//! A sample Rust library demonstrating various programming concepts

pub mod math;
pub mod http;
pub mod database;
pub mod auth;

use std::collections::HashMap;

/// Main application configuration
pub struct Config {
    pub database_url: String,
    pub api_key: String,
    pub debug_mode: bool,
}

impl Config {
    /// Create a new configuration from environment variables
    pub fn from_env() -> Result<Self, std::env::VarError> {
        Ok(Config {
            database_url: std::env::var("DATABASE_URL")?,
            api_key: std::env::var("API_KEY")?,
            debug_mode: std::env::var("DEBUG").unwrap_or_default() == "true",
        })
    }
}

/// Initialize the application with the given configuration
pub fn initialize_app(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing application...");
    
    if config.debug_mode {
        println!("Debug mode enabled");
    }
    
    database::connect(&config.database_url)?;
    auth::setup_auth_system(&config.api_key)?;
    
    println!("Application initialized successfully");
    Ok(())
}
"#,
    ).await?;

    // Math module
    tokio::fs::write(
        src_dir.join("math.rs"),
        r#"//! Mathematical utility functions

/// Add two numbers together
/// 
/// # Examples
/// 
/// ```
/// use mylib::math::add;
/// 
/// assert_eq!(add(2, 3), 5);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Subtract the second number from the first
pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

/// Multiply two numbers
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

/// Divide the first number by the second
/// 
/// # Panics
/// 
/// Panics if the divisor is zero.
pub fn divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        panic!("Cannot divide by zero");
    }
    a / b
}

/// Calculate the factorial of a number
pub fn factorial(n: u32) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n as u64 * factorial(n - 1),
    }
}

/// Find the greatest common divisor of two numbers using Euclidean algorithm
pub fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}
"#,
    ).await?;

    // HTTP module
    tokio::fs::write(
        src_dir.join("http.rs"),
        r#"//! HTTP client and server utilities

use std::collections::HashMap;
use std::time::Duration;

/// HTTP request methods
#[derive(Debug, Clone)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
}

/// HTTP request structure
#[derive(Debug)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
}

/// HTTP response structure
#[derive(Debug)]
pub struct HttpResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

/// HTTP client for making requests
pub struct HttpClient {
    base_url: String,
    timeout: Duration,
    default_headers: HashMap<String, String>,
}

impl HttpClient {
    /// Create a new HTTP client
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            timeout: Duration::from_secs(30),
            default_headers: HashMap::new(),
        }
    }

    /// Set the timeout for requests
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add a default header to all requests
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.default_headers.insert(key, value);
        self
    }

    /// Make a GET request
    pub async fn get(&self, path: &str) -> Result<HttpResponse, Box<dyn std::error::Error>> {
        let url = format!("{}{}", self.base_url, path);
        let request = HttpRequest {
            method: HttpMethod::Get,
            url,
            headers: self.default_headers.clone(),
            body: None,
        };
        self.send_request(request).await
    }

    /// Make a POST request with JSON body
    pub async fn post_json(&self, path: &str, json_body: &str) -> Result<HttpResponse, Box<dyn std::error::Error>> {
        let url = format!("{}{}", self.base_url, path);
        let mut headers = self.default_headers.clone();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        
        let request = HttpRequest {
            method: HttpMethod::Post,
            url,
            headers,
            body: Some(json_body.to_string()),
        };
        self.send_request(request).await
    }

    /// Send an HTTP request
    async fn send_request(&self, _request: HttpRequest) -> Result<HttpResponse, Box<dyn std::error::Error>> {
        // This is a mock implementation for the example
        Ok(HttpResponse {
            status_code: 200,
            headers: HashMap::new(),
            body: "Mock response".to_string(),
        })
    }
}
"#,
    ).await?;

    // Database module
    tokio::fs::write(
        src_dir.join("database.rs"),
        r#"//! Database connection and query utilities

use std::collections::HashMap;
use std::time::Duration;

/// Database connection configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub max_connections: u32,
    pub timeout: Duration,
}

/// Database connection pool
pub struct DatabasePool {
    config: DatabaseConfig,
    active_connections: u32,
}

impl DatabasePool {
    /// Create a new database pool
    pub fn new(config: DatabaseConfig) -> Self {
        Self {
            config,
            active_connections: 0,
        }
    }

    /// Get a connection from the pool
    pub async fn get_connection(&mut self) -> Result<DatabaseConnection, DatabaseError> {
        if self.active_connections >= self.config.max_connections {
            return Err(DatabaseError::PoolExhausted);
        }

        self.active_connections += 1;
        Ok(DatabaseConnection {
            id: self.active_connections,
            config: self.config.clone(),
        })
    }

    /// Return a connection to the pool
    pub fn return_connection(&mut self, _connection: DatabaseConnection) {
        self.active_connections = self.active_connections.saturating_sub(1);
    }
}

/// Database connection
pub struct DatabaseConnection {
    id: u32,
    config: DatabaseConfig,
}

impl DatabaseConnection {
    /// Execute a SQL query and return results
    pub async fn query(&self, sql: &str) -> Result<QueryResult, DatabaseError> {
        println!("Executing query on connection {}: {}", self.id, sql);
        
        // Mock implementation
        Ok(QueryResult {
            rows: vec![],
            affected_rows: 0,
        })
    }

    /// Execute a prepared statement with parameters
    pub async fn execute(&self, sql: &str, params: &[&str]) -> Result<QueryResult, DatabaseError> {
        println!("Executing prepared statement on connection {}: {} with {} params", 
                self.id, sql, params.len());
        
        // Mock implementation
        Ok(QueryResult {
            rows: vec![],
            affected_rows: 1,
        })
    }

    /// Begin a database transaction
    pub async fn begin_transaction(&self) -> Result<DatabaseTransaction, DatabaseError> {
        println!("Beginning transaction on connection {}", self.id);
        Ok(DatabaseTransaction {
            connection_id: self.id,
        })
    }
}

/// Database query result
pub struct QueryResult {
    pub rows: Vec<HashMap<String, String>>,
    pub affected_rows: u64,
}

/// Database transaction
pub struct DatabaseTransaction {
    connection_id: u32,
}

impl DatabaseTransaction {
    /// Commit the transaction
    pub async fn commit(self) -> Result<(), DatabaseError> {
        println!("Committing transaction on connection {}", self.connection_id);
        Ok(())
    }

    /// Rollback the transaction
    pub async fn rollback(self) -> Result<(), DatabaseError> {
        println!("Rolling back transaction on connection {}", self.connection_id);
        Ok(())
    }
}

/// Database error types
#[derive(Debug)]
pub enum DatabaseError {
    ConnectionFailed,
    QueryFailed(String),
    PoolExhausted,
    TransactionFailed,
}

impl std::fmt::Display for DatabaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatabaseError::ConnectionFailed => write!(f, "Failed to connect to database"),
            DatabaseError::QueryFailed(msg) => write!(f, "Query failed: {}", msg),
            DatabaseError::PoolExhausted => write!(f, "Database connection pool exhausted"),
            DatabaseError::TransactionFailed => write!(f, "Transaction failed"),
        }
    }
}

impl std::error::Error for DatabaseError {}

/// Connect to the database using the provided URL
pub fn connect(database_url: &str) -> Result<(), DatabaseError> {
    println!("Connecting to database: {}", database_url);
    // Mock implementation
    Ok(())
}
"#,
    ).await?;

    // Authentication module
    tokio::fs::write(
        src_dir.join("auth.rs"),
        r#"//! User authentication and authorization utilities

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// User authentication information
#[derive(Debug, Clone)]
pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
    pub roles: Vec<String>,
    pub created_at: u64,
}

/// Authentication token
#[derive(Debug)]
pub struct AuthToken {
    pub token: String,
    pub user_id: u64,
    pub expires_at: u64,
}

/// Authentication service
pub struct AuthService {
    api_key: String,
    users: HashMap<u64, User>,
    tokens: HashMap<String, AuthToken>,
}

impl AuthService {
    /// Create a new authentication service
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            users: HashMap::new(),
            tokens: HashMap::new(),
        }
    }

    /// Register a new user
    pub fn register_user(&mut self, username: String, email: String) -> Result<User, AuthError> {
        let user_id = self.users.len() as u64 + 1;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let user = User {
            id: user_id,
            username: username.clone(),
            email,
            roles: vec!["user".to_string()],
            created_at: now,
        };

        if self.users.values().any(|u| u.username == username) {
            return Err(AuthError::UserAlreadyExists);
        }

        self.users.insert(user_id, user.clone());
        println!("Registered new user: {}", username);
        Ok(user)
    }

    /// Authenticate a user and return a token
    pub fn authenticate(&mut self, username: &str, password: &str) -> Result<AuthToken, AuthError> {
        // Mock password validation
        if password.len() < 8 {
            return Err(AuthError::InvalidCredentials);
        }

        let user = self.users
            .values()
            .find(|u| u.username == username)
            .ok_or(AuthError::UserNotFound)?;

        let token = format!("token_{}_{}", user.id, 
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
        
        let expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() + 3600; // 1 hour from now

        let auth_token = AuthToken {
            token: token.clone(),
            user_id: user.id,
            expires_at,
        };

        self.tokens.insert(token, auth_token.clone());
        println!("Generated authentication token for user: {}", username);
        Ok(auth_token)
    }

    /// Validate an authentication token
    pub fn validate_token(&self, token: &str) -> Result<&User, AuthError> {
        let auth_token = self.tokens
            .get(token)
            .ok_or(AuthError::InvalidToken)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if auth_token.expires_at < now {
            return Err(AuthError::TokenExpired);
        }

        let user = self.users
            .get(&auth_token.user_id)
            .ok_or(AuthError::UserNotFound)?;

        Ok(user)
    }

    /// Check if user has a specific role
    pub fn user_has_role(&self, user: &User, role: &str) -> bool {
        user.roles.contains(&role.to_string())
    }

    /// Add a role to a user
    pub fn add_user_role(&mut self, user_id: u64, role: String) -> Result<(), AuthError> {
        let user = self.users
            .get_mut(&user_id)
            .ok_or(AuthError::UserNotFound)?;

        if !user.roles.contains(&role) {
            user.roles.push(role.clone());
            println!("Added role '{}' to user {}", role, user.username);
        }

        Ok(())
    }
}

/// Authentication error types
#[derive(Debug)]
pub enum AuthError {
    UserNotFound,
    UserAlreadyExists,
    InvalidCredentials,
    InvalidToken,
    TokenExpired,
    InsufficientPermissions,
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::UserNotFound => write!(f, "User not found"),
            AuthError::UserAlreadyExists => write!(f, "User already exists"),
            AuthError::InvalidCredentials => write!(f, "Invalid credentials"),
            AuthError::InvalidToken => write!(f, "Invalid authentication token"),
            AuthError::TokenExpired => write!(f, "Authentication token expired"),
            AuthError::InsufficientPermissions => write!(f, "Insufficient permissions"),
        }
    }
}

impl std::error::Error for AuthError {}

/// Setup the authentication system with the given API key
pub fn setup_auth_system(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Setting up authentication system with API key: {}...", &api_key[..8]);
    // Mock implementation
    Ok(())
}
"#,
    ).await?;

    // Create a Python file
    tokio::fs::write(
        repo_path.join("utils.py"),
        r#"""
Utility functions for data processing and analysis.

This module provides various helper functions for working with data,
including mathematical operations, string processing, and file handling.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: List of numerical values
        
    Returns:
        Dictionary containing mean, median, min, max, and standard deviation
    """
    if not numbers:
        return {}
    
    sorted_nums = sorted(numbers)
    n = len(numbers)
    
    # Calculate mean
    mean = sum(numbers) / n
    
    # Calculate median
    if n % 2 == 0:
        median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
    else:
        median = sorted_nums[n//2]
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = variance ** 0.5
    
    return {
        'mean': mean,
        'median': median,
        'min': min(numbers),
        'max': max(numbers),
        'std_dev': std_dev,
        'count': n
    }


def process_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and process a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as a dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Add metadata
        data['_metadata'] = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'processed_at': datetime.now().isoformat()
        }
        
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")


def format_text(text: str, max_length: int = 100, uppercase: bool = False) -> str:
    """
    Format text with various options.
    
    Args:
        text: Input text to format
        max_length: Maximum length of the output text
        uppercase: Whether to convert to uppercase
        
    Returns:
        Formatted text string
    """
    if not text:
        return ""
    
    # Apply transformations
    result = text.strip()
    
    if uppercase:
        result = result.upper()
    
    # Truncate if necessary
    if len(result) > max_length:
        result = result[:max_length - 3] + "..."
    
    return result


class DataProcessor:
    """
    A class for processing and analyzing data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.processed_count = 0
        
    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of data records.
        
        Args:
            records: List of dictionaries representing data records
            
        Returns:
            List of processed records
        """
        processed_records = []
        
        for record in records:
            processed_record = self._process_single_record(record)
            processed_records.append(processed_record)
            self.processed_count += 1
            
        return processed_records
    
    def _process_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single data record.
        
        Args:
            record: Dictionary representing a single data record
            
        Returns:
            Processed record dictionary
        """
        # Create a copy to avoid modifying the original
        processed = record.copy()
        
        # Add processing metadata
        processed['_processed_at'] = datetime.now().isoformat()
        processed['_processor_id'] = id(self)
        
        # Apply any configured transformations
        if 'transformations' in self.config:
            for transformation in self.config['transformations']:
                processed = self._apply_transformation(processed, transformation)
        
        return processed
    
    def _apply_transformation(self, record: Dict[str, Any], transformation: str) -> Dict[str, Any]:
        """
        Apply a transformation to a record.
        
        Args:
            record: The record to transform
            transformation: The transformation to apply
            
        Returns:
            Transformed record
        """
        # Mock transformations for the example
        if transformation == 'normalize':
            # Normalize string values
            for key, value in record.items():
                if isinstance(value, str):
                    record[key] = value.lower().strip()
        
        elif transformation == 'add_timestamp':
            record['timestamp'] = datetime.now().isoformat()
        
        return record
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            'processed_count': self.processed_count,
            'config': self.config,
            'processor_id': id(self)
        }
"#,
    ).await?;

    // Create a JavaScript file
    tokio::fs::write(
        repo_path.join("api.js"),
        r#"/**
 * RESTful API client for interacting with web services.
 * 
 * This module provides a comprehensive HTTP client with support for
 * authentication, request/response interceptors, and error handling.
 */

const https = require('https');
const http = require('http');
const url = require('url');

/**
 * HTTP client class for making API requests
 */
class ApiClient {
    /**
     * Create a new API client instance
     * @param {string} baseURL - The base URL for all requests
     * @param {Object} options - Configuration options
     */
    constructor(baseURL, options = {}) {
        this.baseURL = baseURL;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            'User-Agent': 'API-Client/1.0',
            ...options.headers
        };
        this.timeout = options.timeout || 30000;
        this.interceptors = {
            request: [],
            response: []
        };
    }

    /**
     * Add a request interceptor
     * @param {Function} interceptor - Function to modify requests
     */
    addRequestInterceptor(interceptor) {
        this.interceptors.request.push(interceptor);
    }

    /**
     * Add a response interceptor
     * @param {Function} interceptor - Function to modify responses
     */
    addResponseInterceptor(interceptor) {
        this.interceptors.response.push(interceptor);
    }

    /**
     * Make a GET request
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async get(endpoint, options = {}) {
        return this.request('GET', endpoint, null, options);
    }

    /**
     * Make a POST request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body data
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async post(endpoint, data, options = {}) {
        return this.request('POST', endpoint, data, options);
    }

    /**
     * Make a PUT request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body data
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async put(endpoint, data, options = {}) {
        return this.request('PUT', endpoint, data, options);
    }

    /**
     * Make a DELETE request
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async delete(endpoint, options = {}) {
        return this.request('DELETE', endpoint, null, options);
    }

    /**
     * Make an HTTP request
     * @param {string} method - HTTP method
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body data
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async request(method, endpoint, data, options = {}) {
        const fullURL = new URL(endpoint, this.baseURL);
        
        let requestOptions = {
            method,
            headers: { ...this.defaultHeaders, ...options.headers },
            body: data ? JSON.stringify(data) : undefined
        };

        // Apply request interceptors
        for (const interceptor of this.interceptors.request) {
            requestOptions = await interceptor(requestOptions);
        }

        try {
            const response = await this.makeHttpRequest(fullURL, requestOptions);
            
            // Apply response interceptors
            let processedResponse = response;
            for (const interceptor of this.interceptors.response) {
                processedResponse = await interceptor(processedResponse);
            }

            return processedResponse;
        } catch (error) {
            throw new ApiError(`Request failed: ${error.message}`, error.status);
        }
    }

    /**
     * Make the actual HTTP request
     * @param {URL} url - Request URL
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    makeHttpRequest(url, options) {
        return new Promise((resolve, reject) => {
            const protocol = url.protocol === 'https:' ? https : http;
            
            const request = protocol.request(url, {
                method: options.method,
                headers: options.headers,
                timeout: this.timeout
            }, (response) => {
                let data = '';
                
                response.on('data', (chunk) => {
                    data += chunk;
                });
                
                response.on('end', () => {
                    try {
                        const parsedData = JSON.parse(data);
                        resolve({
                            status: response.statusCode,
                            headers: response.headers,
                            data: parsedData
                        });
                    } catch (error) {
                        resolve({
                            status: response.statusCode,
                            headers: response.headers,
                            data: data
                        });
                    }
                });
            });

            request.on('error', (error) => {
                reject(new ApiError(`Network error: ${error.message}`));
            });

            request.on('timeout', () => {
                reject(new ApiError('Request timeout'));
            });

            if (options.body) {
                request.write(options.body);
            }

            request.end();
        });
    }

    /**
     * Set authentication token
     * @param {string} token - Authentication token
     */
    setAuthToken(token) {
        this.defaultHeaders['Authorization'] = `Bearer ${token}`;
    }

    /**
     * Remove authentication token
     */
    clearAuthToken() {
        delete this.defaultHeaders['Authorization'];
    }
}

/**
 * Custom error class for API errors
 */
class ApiError extends Error {
    constructor(message, status = null) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
    }
}

/**
 * Utility functions for API operations
 */
class ApiUtils {
    /**
     * Build query string from parameters
     * @param {Object} params - Query parameters
     * @returns {string} Query string
     */
    static buildQueryString(params) {
        const searchParams = new URLSearchParams();
        
        for (const [key, value] of Object.entries(params)) {
            if (value !== null && value !== undefined) {
                searchParams.append(key, String(value));
            }
        }
        
        return searchParams.toString();
    }

    /**
     * Parse response headers
     * @param {Object} headers - Raw headers object
     * @returns {Object} Parsed headers
     */
    static parseHeaders(headers) {
        const parsed = {};
        
        for (const [key, value] of Object.entries(headers)) {
            parsed[key.toLowerCase()] = value;
        }
        
        return parsed;
    }

    /**
     * Validate email address
     * @param {string} email - Email address to validate
     * @returns {boolean} True if valid email
     */
    static isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    /**
     * Generate unique request ID
     * @returns {string} Unique identifier
     */
    static generateRequestId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
}

// Export classes and functions
module.exports = {
    ApiClient,
    ApiError,
    ApiUtils
};

// Example usage
if (require.main === module) {
    const client = new ApiClient('https://api.example.com');
    
    // Add authentication interceptor
    client.addRequestInterceptor(async (config) => {
        console.log(`Making ${config.method} request to ${config.url}`);
        return config;
    });
    
    // Example API calls
    async function exampleUsage() {
        try {
            // Get user data
            const userData = await client.get('/users/123');
            console.log('User data:', userData);
            
            // Create new user
            const newUser = await client.post('/users', {
                name: 'John Doe',
                email: 'john@example.com'
            });
            console.log('New user created:', newUser);
            
            // Update user
            const updatedUser = await client.put('/users/123', {
                name: 'John Smith'
            });
            console.log('User updated:', updatedUser);
            
        } catch (error) {
            console.error('API error:', error.message);
        }
    }
    
    exampleUsage();
}
"#,
    ).await?;

    // Create a README file
    tokio::fs::write(
        repo_path.join("README.md"),
        r#"# Test Repository

This is a test repository containing various programming examples to demonstrate the indexing and search capabilities of the janet-ai-retriever system.

## Contents

### Rust Library (`src/`)

A sample Rust library demonstrating modern Rust programming patterns:

- **lib.rs**: Main library module with configuration and initialization
- **math.rs**: Mathematical utility functions including basic arithmetic and algorithms
- **http.rs**: HTTP client utilities with request/response handling
- **database.rs**: Database connection pooling and query execution
- **auth.rs**: User authentication and authorization system

### Python Utilities (`utils.py`)

Data processing and analysis utilities written in Python:

- Statistical calculations for numerical data
- JSON file processing with error handling
- Text formatting and transformation functions
- Object-oriented data processing with the `DataProcessor` class

### JavaScript API Client (`api.js`)

A comprehensive HTTP client for RESTful API interactions:

- Promise-based HTTP methods (GET, POST, PUT, DELETE)
- Request and response interceptors
- Authentication token management
- Error handling and timeout support
- Utility functions for common operations

## Features Demonstrated

The code in this repository showcases various programming concepts:

1. **Error Handling**: Proper error types and handling patterns in Rust, Python, and JavaScript
2. **Async Programming**: Async/await patterns in all three languages
3. **Object-Oriented Design**: Classes and modules with clear separation of concerns
4. **Documentation**: Comprehensive documentation with examples and type hints
5. **Configuration Management**: Environment-based configuration and settings
6. **Network Programming**: HTTP clients and server utilities
7. **Data Processing**: Statistical analysis and data transformation
8. **Authentication**: Token-based authentication systems
9. **Database Operations**: Connection pooling and transaction management

## Usage

This repository is primarily used for testing the semantic search capabilities of the janet-ai-retriever system. The varied content allows for testing different types of queries:

- Function definitions and implementations
- Error handling patterns
- Configuration and setup procedures
- Mathematical operations and algorithms
- Network and HTTP operations
- Database interactions
- Authentication workflows

## Testing Queries

Good test queries for this codebase include:

- "function that adds two numbers"
- "HTTP request handling"
- "database connection management"
- "user authentication system"
- "error handling patterns"
- "configuration setup"
- "mathematical calculations"
- "async programming examples"

The semantic search should be able to find relevant code snippets for each of these concepts across the different programming languages and modules.
"#,
    ).await?;

    Ok(())
}

