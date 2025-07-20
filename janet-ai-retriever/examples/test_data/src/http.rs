//! HTTP client and server utilities

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
    pub async fn post_json(
        &self,
        path: &str,
        json_body: &str,
    ) -> Result<HttpResponse, Box<dyn std::error::Error>> {
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
    async fn send_request(
        &self,
        _request: HttpRequest,
    ) -> Result<HttpResponse, Box<dyn std::error::Error>> {
        // This is a mock implementation for the example
        Ok(HttpResponse {
            status_code: 200,
            headers: HashMap::new(),
            body: "Mock response".to_string(),
        })
    }
}
