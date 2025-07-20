//! User authentication and authorization utilities

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

        let user = self
            .users
            .values()
            .find(|u| u.username == username)
            .ok_or(AuthError::UserNotFound)?;

        let token = format!(
            "token_{}_{}",
            user.id,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        let expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600; // 1 hour from now

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
        let auth_token = self.tokens.get(token).ok_or(AuthError::InvalidToken)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if auth_token.expires_at < now {
            return Err(AuthError::TokenExpired);
        }

        let user = self
            .users
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
        let user = self
            .users
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
    println!(
        "Setting up authentication system with API key: {}...",
        &api_key[..8]
    );
    // Mock implementation
    Ok(())
}
