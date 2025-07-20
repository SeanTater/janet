//! A sample Rust library demonstrating various programming concepts

pub mod auth;
pub mod database;
pub mod http;
pub mod math;

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
