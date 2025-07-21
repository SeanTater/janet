use serde::{Deserialize, Serialize};

/// Network status for external dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    /// Model download connectivity
    pub model_download_connectivity: ConnectivityStatus,
    /// Hugging Face Hub access
    pub hugging_face_hub_access: ConnectivityStatus,
    /// Proxy configuration status
    pub proxy_configuration: ProxyStatus,
    /// SSL certificate validation status
    pub ssl_certificate_validation: bool,
    /// Overall network health
    pub overall_network_health: NetworkHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityStatus {
    /// Is the service reachable?
    pub is_reachable: bool,
    /// Last successful connection timestamp
    pub last_successful_connection: Option<i64>,
    /// Error message if unreachable
    pub error_message: Option<String>,
    /// Response time in milliseconds for last check
    pub response_time_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyStatus {
    /// Is proxy configured?
    pub proxy_configured: bool,
    /// Proxy host and port
    pub proxy_address: Option<String>,
    /// Proxy authentication status
    pub proxy_auth_configured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkHealth {
    Healthy,
    Limited,
    Offline,
}
