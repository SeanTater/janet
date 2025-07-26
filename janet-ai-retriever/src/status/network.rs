use serde::{Deserialize, Serialize};

/// Basic network status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    /// Whether proxy is configured via environment
    pub proxy_configured: bool,
    /// Overall network health assumption
    pub overall_network_health: NetworkHealth,
}

// Removed ConnectivityStatus and ProxyStatus - not tracked

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkHealth {
    Healthy,
    Limited,
    Offline,
}
