//! Compiler utilities.
//!
//! This module contains ID generators

/// Utility struct to generate string IDs with a given prefix.
pub struct IdGenerator {
    prefix: String,
    next_id: i32,
}

impl IdGenerator {
    /// Initialize an IdGenerator that will begin counting up from 0.
    pub fn new(prefix: &str) -> IdGenerator {
        IdGenerator {
            prefix: String::from(prefix),
            next_id: 0,
        }
    }

    /// Generate a new ID.
    pub fn next(&mut self) -> String {
        let res = format!("{}{}", self.prefix, self.next_id);
        self.next_id += 1;
        res
    }
}
