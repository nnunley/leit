// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Postings storage and traversal for Leit retrieval.
//!
//! This crate provides inverted index storage and cursor-based traversal
//! for efficient query execution.

#![no_std]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use leit_core::{EntityId, TermId};

pub mod codec;
pub mod cursor;

/// A single posting: term occurrence in a document with ``Id``.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Posting<Id: EntityId> {
    /// Document containing the term.
    pub doc_id: Id,
    /// Term frequency in this document.
    pub term_freq: u32,
    /// Optional positions (for phrase queries).
    pub positions: Option<Vec<u32>>,
}

/// All postings for a single ``TermId``.
#[derive(Clone, Debug)]
pub struct PostingsList<Id: EntityId> {
    /// The term this list is for.
    pub term_id: TermId,
    /// Postings sorted by `doc_id`.
    pub postings: Vec<Posting<Id>>,
}

impl<Id: EntityId> PostingsList<Id> {
    /// Create a new postings list.
    pub const fn new(term_id: TermId) -> Self {
        Self {
            term_id,
            postings: Vec::new(),
        }
    }

    /// Add a posting.
    pub fn add(&mut self, posting: Posting<Id>) {
        match self
            .postings
            .binary_search_by(|existing| existing.doc_id.cmp(&posting.doc_id))
        {
            Ok(index) => self.postings[index] = posting,
            Err(index) => self.postings.insert(index, posting),
        }
    }

    /// Number of postings.
    pub const fn len(&self) -> usize {
        self.postings.len()
    }

    /// Is empty?
    pub const fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }
}

/// Bidirectional mapping between terms and ``TermId`` values.
#[derive(Debug)]
pub struct InMemoryTermDictionary {
    terms_to_ids: BTreeMap<String, TermId>,
    ids_to_terms: Vec<String>,
    next_id: u32,
}

impl InMemoryTermDictionary {
    /// Create a new term dictionary.
    pub const fn new() -> Self {
        Self {
            terms_to_ids: BTreeMap::new(),
            ids_to_terms: Vec::new(),
            next_id: 0,
        }
    }

    /// Look up a term's ID.
    pub fn lookup(&self, term: &str) -> Option<TermId> {
        self.terms_to_ids.get(term).copied()
    }

    /// Resolve a term ID to its string.
    pub fn resolve(&self, id: TermId) -> Option<&str> {
        self.ids_to_terms
            .get(id.as_u32() as usize)
            .map(String::as_str)
    }

    /// Insert a term, returning its ID.
    pub fn insert(&mut self, term: &str) -> TermId {
        if let Some(&id) = self.terms_to_ids.get(term) {
            return id;
        }
        let id = TermId::new(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("term dictionary capacity exhausted");
        self.terms_to_ids.insert(term.to_string(), id);
        self.ids_to_terms.push(term.to_string());
        id
    }

    /// Number of terms.
    pub const fn len(&self) -> usize {
        self.ids_to_terms.len()
    }

    /// Is empty?
    pub const fn is_empty(&self) -> bool {
        self.ids_to_terms.is_empty()
    }
}

impl Default for InMemoryTermDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// Cursor for traversing postings by document.
pub trait DocCursor<Id: EntityId> {
    /// Get the current document ID.
    fn doc(&self) -> Option<Id>;

    /// Advance to the next document.
    /// Returns true if there is a next document.
    fn advance(&mut self) -> bool;

    /// Seek to a specific document or the first after it.
    /// Returns true if found.
    fn seek(&mut self, target: Id) -> bool;
}

/// Cursor with term frequency access.
pub trait TfCursor<Id: EntityId>: DocCursor<Id> {
    /// Get term frequency for the current document.
    fn term_freq(&self) -> u32;
}

/// Public block-aware cursor state for optional pruning-oriented traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockCursorState<Id: EntityId> {
    /// The cursor does not expose block metadata.
    Unsupported,
    /// The cursor has advanced past the end of the postings list.
    Exhausted,
    /// The cursor is positioned on a block with simple summary metadata.
    Ready {
        /// Inclusive end document for the current block.
        end_doc: Id,
        /// Maximum term frequency within the current block.
        max_term_freq: u32,
    },
}

/// Optional block-aware cursor extension.
pub trait BlockCursor<Id: EntityId>: TfCursor<Id> {
    /// Return the current block summary or explain why it is unavailable.
    fn block_state(&self) -> BlockCursorState<Id>;

    /// Advance to the next block.
    /// Returns `true` when another block is available.
    fn advance_block(&mut self) -> bool;
}

/// In-memory postings index.
#[derive(Debug)]
pub struct InMemoryPostings<Id: EntityId> {
    #[expect(dead_code, reason = "reserved for term resolution")]
    term_dict: InMemoryTermDictionary,
    postings: BTreeMap<TermId, PostingsList<Id>>,
}

impl<Id: EntityId> InMemoryPostings<Id> {
    /// Create a new in-memory postings index.
    pub const fn new() -> Self {
        Self {
            term_dict: InMemoryTermDictionary::new(),
            postings: BTreeMap::new(),
        }
    }

    /// Add a postings list.
    pub fn add(&mut self, list: PostingsList<Id>) {
        self.postings.insert(list.term_id, list);
    }

    /// Get a cursor for a term.
    pub fn cursor(&self, term_id: TermId) -> Option<InMemoryCursor<'_, Id>> {
        self.postings.get(&term_id).map(|list| InMemoryCursor {
            postings: &list.postings,
            pos: 0,
        })
    }

    /// Number of terms.
    pub fn len(&self) -> usize {
        self.postings.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }
}

impl<Id: EntityId> Default for InMemoryPostings<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// Cursor over in-memory postings.
#[derive(Debug)]
pub struct InMemoryCursor<'a, Id: EntityId> {
    postings: &'a [Posting<Id>],
    pos: usize,
}

impl<Id: EntityId> DocCursor<Id> for InMemoryCursor<'_, Id> {
    fn doc(&self) -> Option<Id> {
        self.postings.get(self.pos).map(|p| p.doc_id)
    }

    fn advance(&mut self) -> bool {
        if self.pos < self.postings.len() {
            self.pos += 1;
        }
        self.pos < self.postings.len()
    }

    fn seek(&mut self, target: Id) -> bool {
        if self.pos >= self.postings.len() {
            return false;
        }
        let remaining = &self.postings[self.pos..];
        match remaining.binary_search_by(|p| p.doc_id.cmp(&target)) {
            Ok(offset) => {
                self.pos = self.pos.saturating_add(offset);
                true
            }
            Err(offset) => {
                self.pos = self.pos.saturating_add(offset);
                self.pos < self.postings.len()
            }
        }
    }
}

impl<Id: EntityId> TfCursor<Id> for InMemoryCursor<'_, Id> {
    fn term_freq(&self) -> u32 {
        self.postings.get(self.pos).map_or(0, |p| p.term_freq)
    }
}

impl<Id: EntityId> BlockCursor<Id> for InMemoryCursor<'_, Id> {
    fn block_state(&self) -> BlockCursorState<Id> {
        self.postings
            .get(self.pos)
            .map_or(BlockCursorState::Exhausted, |posting| {
                BlockCursorState::Ready {
                    end_doc: posting.doc_id,
                    max_term_freq: posting.term_freq,
                }
            })
    }

    fn advance_block(&mut self) -> bool {
        self.advance()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_term_dictionary() {
        let mut dict = InMemoryTermDictionary::new();
        let id1 = dict.insert("hello");
        let id2 = dict.insert("world");
        let id3 = dict.insert("hello"); // Duplicate

        assert_eq!(id1, id3); // Same term, same ID
        assert_ne!(id1, id2);
        assert_eq!(dict.lookup("hello"), Some(id1));
        assert_eq!(dict.resolve(id2), Some("world"));
    }

    #[test]
    fn test_in_memory_cursor() {
        let mut postings = InMemoryPostings::<u32>::new();
        let term_id = TermId::new(0);
        let mut list = PostingsList::new(term_id);
        list.add(Posting {
            doc_id: 1_u32,
            term_freq: 2,
            positions: None,
        });
        list.add(Posting {
            doc_id: 3_u32,
            term_freq: 1,
            positions: None,
        });
        list.add(Posting {
            doc_id: 5_u32,
            term_freq: 3,
            positions: None,
        });
        postings.add(list);

        let mut cursor = postings.cursor(term_id).unwrap();
        assert_eq!(cursor.doc(), Some(1_u32));
        assert_eq!(cursor.term_freq(), 2);
        assert_eq!(
            cursor.block_state(),
            BlockCursorState::Ready {
                end_doc: 1_u32,
                max_term_freq: 2,
            }
        );

        assert!(cursor.advance());
        assert_eq!(cursor.doc(), Some(3_u32));
        assert_eq!(
            cursor.block_state(),
            BlockCursorState::Ready {
                end_doc: 3_u32,
                max_term_freq: 1,
            }
        );

        assert!(cursor.seek(4_u32));
        assert_eq!(cursor.doc(), Some(5_u32));
        assert_eq!(
            cursor.block_state(),
            BlockCursorState::Ready {
                end_doc: 5_u32,
                max_term_freq: 3,
            }
        );
    }

    #[test]
    fn test_postings_list_keeps_doc_ids_sorted() {
        let term_id = TermId::new(0);
        let mut list = PostingsList::new(term_id);
        list.add(Posting {
            doc_id: 5_u32,
            term_freq: 1,
            positions: None,
        });
        list.add(Posting {
            doc_id: 1_u32,
            term_freq: 2,
            positions: None,
        });
        list.add(Posting {
            doc_id: 3_u32,
            term_freq: 4,
            positions: None,
        });

        let doc_ids: Vec<_> = list.postings.iter().map(|posting| posting.doc_id).collect();
        assert_eq!(doc_ids, vec![1, 3, 5]);

        let mut postings = InMemoryPostings::<u32>::new();
        postings.add(list);

        let mut cursor = postings.cursor(term_id).expect("cursor should exist");
        assert_eq!(cursor.doc(), Some(1));
        assert!(cursor.seek(4));
        assert_eq!(cursor.doc(), Some(5));
    }
}
