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

pub fn get_cc() -> String {
  match env::var("CC") {
      Ok(val) => val,
      Err(_) => "ncc".to_string(),
  }
}

pub fn get_cflags() -> String {
  match env::var("WELD_CFLAGS") {
      Ok(val) => val,
      Err(_) => get_ncc_cflags(),
  }
}

// Old environment options
pub fn get_ncc_cflags() -> String {
  match env::var("WELD_NCC_CFLAGS") {
      Ok(val) => val,
      Err(_) => "".to_string(),
  }
}

pub fn get_veweld_cflags() -> String {
  match env::var("VEWELD_CFLAGS") {
      Ok(val) => val,
      Err(_) => "".to_string(),
  }
}

pub fn get_veweld_conv_bool_to_int() -> bool {
  let veweld_cflags = get_veweld_cflags();
  !(veweld_cflags.contains("-veweld-no-conv-bool-to-int"))
}

pub fn get_veweld_infer_filter_size() -> bool {
  let veweld_cflags = get_veweld_cflags();
  veweld_cflags.contains("-veweld-infer-filter-size")
}

pub fn get_home() -> String {
  match env::var("WELD_HOME") {
      Ok(val) => val,
      Err(_) => "".to_string(),
  }
}
