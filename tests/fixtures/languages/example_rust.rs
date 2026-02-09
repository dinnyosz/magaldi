//! Example Rust file exercising all extractable element types.
//!
//! This fixture ensures the magaldi index always contains Rust-specific
//! elements (trait, enum, struct/class, impl, generics) when parsing its own repo.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

// Constants
const MAX_RETRIES: u32 = 3;
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const BUFFER_SIZE: usize = 8192;

// Static variables
static GLOBAL_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// --- Enums ---

#[derive(Debug, Clone, PartialEq)]
enum Status {
    Pending,
    Active,
    Suspended { reason: String },
    Deleted,
}

#[derive(Debug, Clone, Copy)]
enum Priority {
    Low = 1,
    Medium = 5,
    High = 10,
    Critical = 100,
}

#[derive(Debug)]
enum AppError {
    NotFound(String),
    Validation(String),
    Internal(Box<dyn std::error::Error>),
    Timeout { elapsed: u64, limit: u64 },
}

// --- Traits ---

trait Repository<T> {
    fn find_by_id(&self, id: &str) -> Option<&T>;
    fn find_all(&self) -> Vec<&T>;
    fn save(&mut self, entity: T) -> Result<(), AppError>;
    fn delete(&mut self, id: &str) -> Result<bool, AppError>;
}

trait Serializable {
    fn serialize(&self) -> Vec<u8>;
    fn deserialize(data: &[u8]) -> Result<Self, AppError>
    where
        Self: Sized;
}

trait Transformer {
    type Input;
    type Output;

    fn transform(&self, input: Self::Input) -> Result<Self::Output, AppError>;
    fn supports(&self, input: &Self::Input) -> bool;
}

trait EventHandler: Send + Sync {
    fn handle(&self, event: &str, data: &[u8]);
    fn subscribed_events(&self) -> Vec<String>;
}

// --- Structs (extracted as class) ---

#[derive(Debug, Clone)]
struct Config {
    name: String,
    timeout: u64,
    retries: u32,
    tags: Vec<String>,
}

#[derive(Debug)]
struct User {
    id: String,
    name: String,
    email: String,
    status: Status,
    priority: Priority,
}

struct InMemoryRepo<T> {
    store: HashMap<String, T>,
}

struct Cache<V> {
    entries: HashMap<String, CacheEntry<V>>,
    max_size: usize,
}

struct CacheEntry<V> {
    value: V,
    expiry: u64,
    tags: Vec<String>,
}

// --- Implementations ---

impl Config {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            timeout: DEFAULT_TIMEOUT_SECS,
            retries: MAX_RETRIES,
            tags: Vec::new(),
        }
    }

    fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = timeout;
        self
    }

    fn validate(&self) -> Result<(), AppError> {
        if self.name.is_empty() {
            return Err(AppError::Validation("Name cannot be empty".into()));
        }
        if self.timeout == 0 {
            return Err(AppError::Validation("Timeout must be positive".into()));
        }
        Ok(())
    }
}

impl User {
    fn new(name: &str, email: &str) -> Self {
        let id = format!("user_{}", GLOBAL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
        Self {
            id,
            name: name.to_string(),
            email: email.to_string(),
            status: Status::Pending,
            priority: Priority::Medium,
        }
    }

    fn activate(&mut self) {
        self.status = Status::Active;
    }

    fn suspend(&mut self, reason: &str) {
        self.status = Status::Suspended {
            reason: reason.to_string(),
        };
    }

    fn is_active(&self) -> bool {
        matches!(self.status, Status::Active)
    }
}

impl fmt::Display for User {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "User({}: {})", self.id, self.name)
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::NotFound(id) => write!(f, "Not found: {}", id),
            AppError::Validation(msg) => write!(f, "Validation error: {}", msg),
            AppError::Internal(e) => write!(f, "Internal error: {}", e),
            AppError::Timeout { elapsed, limit } => {
                write!(f, "Timeout: {}s elapsed (limit: {}s)", elapsed, limit)
            }
        }
    }
}

impl<T: Clone> InMemoryRepo<T> {
    fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }
}

impl Repository<User> for InMemoryRepo<User> {
    fn find_by_id(&self, id: &str) -> Option<&User> {
        self.store.get(id)
    }

    fn find_all(&self) -> Vec<&User> {
        self.store.values().collect()
    }

    fn save(&mut self, entity: User) -> Result<(), AppError> {
        self.store.insert(entity.id.clone(), entity);
        Ok(())
    }

    fn delete(&mut self, id: &str) -> Result<bool, AppError> {
        Ok(self.store.remove(id).is_some())
    }
}

impl<V> Cache<V> {
    fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_size,
        }
    }

    fn get(&self, key: &str) -> Option<&V> {
        self.entries.get(key).map(|e| &e.value)
    }

    fn set(&mut self, key: String, value: V, ttl: u64, tags: Vec<String>) {
        if self.entries.len() >= self.max_size {
            // Evict oldest (simplified)
            if let Some(first_key) = self.entries.keys().next().cloned() {
                self.entries.remove(&first_key);
            }
        }
        self.entries.insert(
            key,
            CacheEntry {
                value,
                expiry: ttl,
                tags,
            },
        );
    }
}

// --- Free Functions ---

fn parse_config(path: &str) -> Result<Config, AppError> {
    if path.is_empty() {
        return Err(AppError::Validation("Path cannot be empty".into()));
    }
    Ok(Config::new(path))
}

fn validate_email(email: &str) -> bool {
    email.contains('@') && email.contains('.')
}

fn retry<F, T>(f: F, attempts: u32) -> Result<T, AppError>
where
    F: Fn() -> Result<T, AppError>,
{
    let mut last_error = None;
    for _ in 0..attempts {
        match f() {
            Ok(value) => return Ok(value),
            Err(e) => last_error = Some(e),
        }
    }
    Err(last_error.unwrap_or(AppError::Internal("Exhausted retries".into())))
}

fn process_batch<T, F>(items: Vec<T>, processor: F) -> Vec<Result<T, AppError>>
where
    F: Fn(T) -> Result<T, AppError>,
{
    items.into_iter().map(processor).collect()
}
