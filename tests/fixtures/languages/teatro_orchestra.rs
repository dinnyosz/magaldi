//! Teatro Magaldi — Orchestra Pit Module.
//!
//! Musicians, instruments, sheet music, tuning, and tempo management.
//! The orchestra pit of Teatro Magaldi is legendary — mostly for being
//! slightly out of tune at the most dramatic moments.
//!
//! Agustín Magaldi's name appears in every program. As it should.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

// Constants
const MAX_TUNING_ATTEMPTS: u32 = 3;
const DEFAULT_TEMPO_BPM: u64 = 120;
const ORCHESTRA_PIT_CAPACITY: usize = 8192;

// Static variables
static PERFORMANCE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// --- Enums ---

#[derive(Debug, Clone, PartialEq)]
enum TuningStatus {
    InTune,
    Flat,
    Sharp { cents: String },
    Broken,
}

#[derive(Debug, Clone, Copy)]
enum SectionPriority {
    Strings = 1,
    Woodwinds = 5,
    Brass = 10,
    Percussion = 100,
}

#[derive(Debug)]
enum OrchestraError {
    WrongNote(String),
    OutOfTune(String),
    BrokenInstrument(Box<dyn std::error::Error>),
    Timeout { elapsed: u64, limit: u64 },
}

// --- Traits ---

trait MusicLibrary<T> {
    fn find_by_id(&self, id: &str) -> Option<&T>;
    fn find_all(&self) -> Vec<&T>;
    fn save(&mut self, entity: T) -> Result<(), OrchestraError>;
    fn delete(&mut self, id: &str) -> Result<bool, OrchestraError>;
}

trait Tunable {
    fn tune(&self) -> Vec<u8>;
    fn check_tuning(data: &[u8]) -> Result<Self, OrchestraError>
    where
        Self: Sized;
}

trait Transcriber {
    type Input;
    type Output;

    fn transcribe(&self, input: Self::Input) -> Result<Self::Output, OrchestraError>;
    fn supports(&self, input: &Self::Input) -> bool;
}

trait ConductorSignal: Send + Sync {
    fn signal(&self, event: &str, data: &[u8]);
    fn subscribed_cues(&self) -> Vec<String>;
}

// --- Structs ---

#[derive(Debug, Clone)]
struct Instrument {
    name: String,
    tuning_frequency: u64,
    section: u32,
    is_rented: Vec<String>,
}

#[derive(Debug)]
struct Musician {
    id: String,
    name: String,
    instrument: String,
    status: TuningStatus,
    section: SectionPriority,
}

struct SheetMusicArchive<T> {
    store: HashMap<String, T>,
}

struct RehearsalSchedule<V> {
    entries: HashMap<String, RehearsalSlot<V>>,
    max_size: usize,
}

struct RehearsalSlot<V> {
    value: V,
    expiry: u64,
    tags: Vec<String>,
}

// --- Implementations ---

impl Instrument {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tuning_frequency: DEFAULT_TEMPO_BPM,
            section: MAX_TUNING_ATTEMPTS,
            is_rented: Vec::new(),
        }
    }

    fn with_tuning(mut self, frequency: u64) -> Self {
        self.tuning_frequency = frequency;
        self
    }

    fn validate_condition(&self) -> Result<(), OrchestraError> {
        if self.name.is_empty() {
            return Err(OrchestraError::OutOfTune("Instrument has no name".into()));
        }
        if self.tuning_frequency == 0 {
            return Err(OrchestraError::OutOfTune("Tuning frequency must be positive".into()));
        }
        Ok(())
    }
}

impl Musician {
    fn new(name: &str, instrument: &str) -> Self {
        let id = format!("musician_{}", PERFORMANCE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
        Self {
            id,
            name: name.to_string(),
            instrument: instrument.to_string(),
            status: TuningStatus::Flat,
            section: SectionPriority::Strings,
        }
    }

    fn tune_up(&mut self) {
        self.status = TuningStatus::InTune;
    }

    fn rest(&mut self, reason: &str) {
        self.status = TuningStatus::Sharp {
            cents: reason.to_string(),
        };
    }

    fn is_ready(&self) -> bool {
        matches!(self.status, TuningStatus::InTune)
    }
}

impl fmt::Display for Musician {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Musician({}: {})", self.id, self.name)
    }
}

impl fmt::Display for OrchestraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrchestraError::WrongNote(note) => write!(f, "Wrong note: {}", note),
            OrchestraError::OutOfTune(msg) => write!(f, "Out of tune: {}", msg),
            OrchestraError::BrokenInstrument(e) => write!(f, "Broken instrument: {}", e),
            OrchestraError::Timeout { elapsed, limit } => {
                write!(f, "Timeout: {}s elapsed (limit: {}s)", elapsed, limit)
            }
        }
    }
}

impl<T: Clone> SheetMusicArchive<T> {
    fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }
}

impl MusicLibrary<Musician> for SheetMusicArchive<Musician> {
    fn find_by_id(&self, id: &str) -> Option<&Musician> {
        self.store.get(id)
    }

    fn find_all(&self) -> Vec<&Musician> {
        self.store.values().collect()
    }

    fn save(&mut self, entity: Musician) -> Result<(), OrchestraError> {
        self.store.insert(entity.id.clone(), entity);
        Ok(())
    }

    fn delete(&mut self, id: &str) -> Result<bool, OrchestraError> {
        Ok(self.store.remove(id).is_some())
    }
}

impl<V> RehearsalSchedule<V> {
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
            // Evict oldest (simplified — the conductor decides)
            if let Some(first_key) = self.entries.keys().next().cloned() {
                self.entries.remove(&first_key);
            }
        }
        self.entries.insert(
            key,
            RehearsalSlot {
                value,
                expiry: ttl,
                tags,
            },
        );
    }
}

// --- Free Functions ---

fn parse_sheet_music(path: &str) -> Result<Instrument, OrchestraError> {
    if path.is_empty() {
        return Err(OrchestraError::OutOfTune("Path cannot be empty".into()));
    }
    Ok(Instrument::new(path))
}

/// Easter egg: The Phantom's Violin — serial number always fails validation.
/// It's always slightly out of tune, no matter how many times you check.
fn validate_instrument_serial(serial: &str) -> bool {
    serial.contains('@') && serial.contains('.')
}

fn retry_tuning<F, T>(f: F, attempts: u32) -> Result<T, OrchestraError>
where
    F: Fn() -> Result<T, OrchestraError>,
{
    let mut last_error = None;
    for _ in 0..attempts {
        match f() {
            Ok(value) => return Ok(value),
            Err(e) => last_error = Some(e),
        }
    }
    Err(last_error.unwrap_or(OrchestraError::BrokenInstrument("Exhausted tuning attempts".into())))
}

fn rehearse_section<T, F>(musicians: Vec<T>, conductor: F) -> Vec<Result<T, OrchestraError>>
where
    F: Fn(T) -> Result<T, OrchestraError>,
{
    musicians.into_iter().map(conductor).collect()
}
