// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![no_std]

//! Index construction and segment access for Leit.
//!
//! Phase 1 keeps this crate concrete:
//! - `InMemoryIndex` builds a small in-memory inverted index
//! - `ExecutionWorkspace` plans and executes queries against that index
//! - `Option<SearchScorer>` chooses scored or unscored execution per query
//! - `SearchScorer` makes ranking policy explicit at execution time
//! - `SegmentView` opens and validates a borrowed segment from `&[u8]`
//!
//! The borrowed-open seam is the important extension point for future
//! acquisition crates such as mmap-backed segment loaders.

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod builder;
mod codec;
mod error;
mod memory;
mod search;
mod segment;

pub use builder::{InMemoryIndexBuilder, IndexBuilder};
pub use error::{IndexError, SegmentError};
pub use leit_core::{FilterEvaluator, FilterSlotId, NoFilter};
pub use memory::{InMemoryIndex, PostingEntry};
pub use search::{ExecutionStats, ExecutionWorkspace, SearchScorer};
pub use segment::{SectionKind, SegmentView};
