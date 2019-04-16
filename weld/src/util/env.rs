//! Utility functions to check environment variables

use std::env;

pub fn get_weld_compilation_stats() -> bool {
  match env::var("WELD_COMPILATION_STATS") {
      Ok(val) => val,
      Err(_) => "false".to_string(),
  }.parse().unwrap_or(false)
}

pub fn get_weld_run_stats() -> bool {
  match env::var("WELD_RUN_STATS") {
      Ok(val) => val,
      Err(_) => "false".to_string(),
  }.parse().unwrap_or(false)
}
