// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Zero-copy in-memory cursor implementation for postings.
//!
//! `MemPostingsCursor` provides traversal over uncompressed postings stored in memory,
//! implementing the cursor traits defined in `leit_postings::cursor`. It requires no
//! encoding, decoding, or allocation — just a borrowed slice of `PostingEntry` values.

use crate::memory::PostingEntry;
use leit_postings::cursor::{CursorStatus, DocCursor, TfCursor};

/// A zero-copy cursor over an in-memory postings slice.
///
/// Wraps `&'a [PostingEntry]` and a current position. Traversal does not require
/// encoding/decoding (unlike `CompressedCursor`) and is allocation-free.
/// Implements both `DocCursor` and `TfCursor` for full-featured cursor API compatibility.
#[derive(Debug)]
pub(crate) struct MemPostingsCursor<'a> {
    /// The borrowed postings slice (doc-sorted ascending).
    postings: &'a [PostingEntry],
    /// Current position in the postings slice.
    pos: usize,
}

impl<'a> MemPostingsCursor<'a> {
    /// Create a new cursor positioned at the start of the postings slice.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "constructed by the execution path in ITER-0003B T2"
        )
    )]
    pub(crate) fn new(postings: &'a [PostingEntry]) -> Self {
        Self { postings, pos: 0 }
    }
}

impl<'a> DocCursor for MemPostingsCursor<'a> {
    fn current_doc(&self) -> Option<u32> {
        self.postings.get(self.pos).map(|p| p.doc_id)
    }

    fn advance(&mut self) -> CursorStatus {
        // Increment position; return Valid if a doc is now available, else Exhausted.
        if self.pos < self.postings.len() {
            self.pos += 1;
        }
        if self.pos < self.postings.len() {
            CursorStatus::Valid
        } else {
            CursorStatus::Exhausted
        }
    }

    fn advance_to(&mut self, target: u32) -> CursorStatus {
        // Binary search over remaining postings to find first doc >= target.
        // Never move backward.
        if self.pos >= self.postings.len() {
            return CursorStatus::Exhausted;
        }

        let remaining = &self.postings[self.pos..];
        match remaining.binary_search_by_key(&target, |p| p.doc_id) {
            Ok(offset) => {
                // Exact match: found a posting with doc_id == target.
                self.pos += offset;
                CursorStatus::Valid
            }
            Err(offset) => {
                // Insert position: first doc > target, or end of slice.
                self.pos += offset;
                if self.pos < self.postings.len() {
                    CursorStatus::Valid
                } else {
                    CursorStatus::Exhausted
                }
            }
        }
    }
}

impl<'a> TfCursor for MemPostingsCursor<'a> {
    fn current_tf(&self) -> u32 {
        // Return term frequency for current posting; undefined if exhausted.
        self.postings
            .get(self.pos)
            .map(|p| p.term_freq)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_postings() {
        let postings = [];
        let cursor = MemPostingsCursor::new(&postings);
        assert_eq!(cursor.current_doc(), None);
    }

    #[test]
    fn test_single_posting() {
        let postings = [PostingEntry {
            doc_id: 42,
            term_freq: 3,
        }];
        let mut cursor = MemPostingsCursor::new(&postings);

        // Initial position: should be at first posting.
        assert_eq!(cursor.current_doc(), Some(42));
        assert_eq!(cursor.current_tf(), 3);

        // Advance from first posting.
        let status = cursor.advance();
        assert_eq!(status, CursorStatus::Exhausted);
        assert_eq!(cursor.current_doc(), None);
    }

    #[test]
    fn test_sequential_advance() {
        let postings = [
            PostingEntry {
                doc_id: 10,
                term_freq: 2,
            },
            PostingEntry {
                doc_id: 20,
                term_freq: 5,
            },
            PostingEntry {
                doc_id: 30,
                term_freq: 1,
            },
        ];
        let mut cursor = MemPostingsCursor::new(&postings);

        // Position 0: doc_id=10.
        assert_eq!(cursor.current_doc(), Some(10));
        assert_eq!(cursor.current_tf(), 2);

        // Advance to position 1: doc_id=20.
        assert_eq!(cursor.advance(), CursorStatus::Valid);
        assert_eq!(cursor.current_doc(), Some(20));
        assert_eq!(cursor.current_tf(), 5);

        // Advance to position 2: doc_id=30.
        assert_eq!(cursor.advance(), CursorStatus::Valid);
        assert_eq!(cursor.current_doc(), Some(30));
        assert_eq!(cursor.current_tf(), 1);

        // Advance past end: exhausted.
        assert_eq!(cursor.advance(), CursorStatus::Exhausted);
        assert_eq!(cursor.current_doc(), None);
    }

    #[test]
    fn test_advance_to_exact_match() {
        let postings = [
            PostingEntry {
                doc_id: 10,
                term_freq: 2,
            },
            PostingEntry {
                doc_id: 20,
                term_freq: 5,
            },
            PostingEntry {
                doc_id: 30,
                term_freq: 1,
            },
        ];
        let mut cursor = MemPostingsCursor::new(&postings);

        // Seek to doc_id=20 from start.
        let status = cursor.advance_to(20);
        assert_eq!(status, CursorStatus::Valid);
        assert_eq!(cursor.current_doc(), Some(20));
        assert_eq!(cursor.current_tf(), 5);
    }

    #[test]
    fn test_advance_to_no_exact_match() {
        let postings = [
            PostingEntry {
                doc_id: 10,
                term_freq: 2,
            },
            PostingEntry {
                doc_id: 20,
                term_freq: 5,
            },
            PostingEntry {
                doc_id: 30,
                term_freq: 1,
            },
        ];
        let mut cursor = MemPostingsCursor::new(&postings);

        // Seek to doc_id=15 (between 10 and 20) — should land on 20.
        let status = cursor.advance_to(15);
        assert_eq!(status, CursorStatus::Valid);
        assert_eq!(cursor.current_doc(), Some(20));
        assert_eq!(cursor.current_tf(), 5);
    }

    #[test]
    fn test_advance_to_past_end() {
        let postings = [
            PostingEntry {
                doc_id: 10,
                term_freq: 2,
            },
            PostingEntry {
                doc_id: 20,
                term_freq: 5,
            },
        ];
        let mut cursor = MemPostingsCursor::new(&postings);

        // Seek to doc_id=100 (past all postings) — should be exhausted.
        let status = cursor.advance_to(100);
        assert_eq!(status, CursorStatus::Exhausted);
        assert_eq!(cursor.current_doc(), None);
    }

    #[test]
    fn test_advance_to_before_first() {
        let postings = [
            PostingEntry {
                doc_id: 10,
                term_freq: 2,
            },
            PostingEntry {
                doc_id: 20,
                term_freq: 5,
            },
        ];
        let mut cursor = MemPostingsCursor::new(&postings);

        // Seek to doc_id=5 (before all) — should land on 10.
        let status = cursor.advance_to(5);
        assert_eq!(status, CursorStatus::Valid);
        assert_eq!(cursor.current_doc(), Some(10));
        assert_eq!(cursor.current_tf(), 2);
    }

    #[test]
    fn test_no_backward_movement() {
        let postings = [
            PostingEntry {
                doc_id: 10,
                term_freq: 2,
            },
            PostingEntry {
                doc_id: 20,
                term_freq: 5,
            },
            PostingEntry {
                doc_id: 30,
                term_freq: 1,
            },
        ];
        let mut cursor = MemPostingsCursor::new(&postings);

        // Advance to doc_id=20.
        cursor.advance_to(20);
        assert_eq!(cursor.current_doc(), Some(20));

        // Attempt to seek backward to doc_id=10 — should remain at 20 (no backward movement).
        let status = cursor.advance_to(10);
        assert_eq!(status, CursorStatus::Valid);
        assert_eq!(cursor.current_doc(), Some(20));
    }

    #[test]
    fn test_advance_to_on_exhausted() {
        let postings = [PostingEntry {
            doc_id: 10,
            term_freq: 2,
        }];
        let mut cursor = MemPostingsCursor::new(&postings);

        // Exhaust the cursor.
        cursor.advance();
        assert_eq!(cursor.current_doc(), None);

        // Attempt advance_to on exhausted cursor — should remain exhausted.
        let status = cursor.advance_to(5);
        assert_eq!(status, CursorStatus::Exhausted);
    }

    #[test]
    fn test_repeated_advance_after_exhaustion_is_idempotent() {
        let postings = [PostingEntry {
            doc_id: 10,
            term_freq: 2,
        }];
        let mut cursor = MemPostingsCursor::new(&postings);

        // First advance exhausts the single-element cursor.
        assert_eq!(cursor.advance(), CursorStatus::Exhausted);
        assert_eq!(cursor.current_doc(), None);

        // Further advances stay Exhausted without moving past the end or panicking
        // (matches CompressedCursor: position is clamped, never overflows).
        assert_eq!(cursor.advance(), CursorStatus::Exhausted);
        assert_eq!(cursor.advance(), CursorStatus::Exhausted);
        assert_eq!(cursor.current_doc(), None);
    }
}
