//! Utility functions to perform offload call for VE

use std::env;

pub fn get_ve_node_number() -> i64 {
  match env::var("VE_NODE_NUMBER") {
      Ok(val) => val,
      Err(_) => "0".to_string(),
  }.parse().unwrap_or(0)
}

#[cfg(feature = "offload_ve")]
pub fn offload_ve() -> bool {
  let node = get_ve_node_number();
  if node >= 0 {
    // any number, un-parsable string, or empty
    true
  } else {
    // explicitly defined minus number
    false
  }
}

#[cfg(not(feature = "offload_ve"))]
pub fn offload_ve() -> bool {
  false
}
