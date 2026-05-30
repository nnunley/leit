// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cursor-based traversal API for compressed postings.
//!
//! This module provides zero-allocation, workspace-borrowed cursor traits and
//! implementations for iterating over compressed postings, with optional block-aware
//! capabilities for Phase 3 skip-list pruning.
//!
//! ## Architecture
//!
//! Cursors are built on top of the codec layer (`DeltaVarint`, `BlockDelta`). The codec
//! encodes postings into compressed bytes; the cursor layer decodes and traverses them
//! with a borrowed `DecodeScratch` for zero-allocation steady-state performance.
//!
//! ### Public API (traversal traits)
//!
//! - `DocCursor` — base trait for document traversal (`current_doc`, `advance`, `advance_to`).
//! - `TfCursor: DocCursor` — extends with term frequency (`current_tf`).
//! - `BlockCursor: DocCursor` — extends with block-aware metadata (`block_end_doc`, `block_max_score`).
//!
//! Layering keeps the hot path (doc-only or doc+TF) lean and allocation-free, and enables
//! scorers to request exactly the capability they need at the type level (DEC-14).
//!
//! ### Internal (cursor implementations)
//!
//! `CompressedCursor<'a>` is the canonical implementation, powered by codec decode and
//! stepping over decoded slices. Block-aware metadata is exposed via the `BlockCursor`
//! trait; **pruning/skip execution is Phase 3 and NOT included in the v1 API surface**.
//!
//! ## Public / Internal Boundary
//!
//! **Public API:**
//! - Trait definitions: `DocCursor`, `TfCursor`, `BlockCursor`, `CursorFactory`, `BlockDecoder`.
//! - Core types: `CursorStatus`, `DecodeScratch`, `BlockSummary`, `PostingsView`, `ScoreBound`.
//! - Factory: `DefaultCursorFactory`.
//! - Concrete cursor: `CompressedCursor` (for trait impl visibility; construct via factory).
//!
//! **Internal/Crate-private:**
//! - Block iteration logic (block index computation, position tracking).
//! - Pruning/skip execution (deferred to Phase 3; see `BlockCursor` docs).
//! - Cursor traversal state (position in decoded slices).
//!
//! ## Forward extensibility (STORY-0010)
//!
//! The layout is designed to accommodate future posting variants without breaking the v1
//! cursor API:
//! - **Positions / payloads** — `DocCursor`/`TfCursor` are the v1 capabilities; a future
//!   `PositionCursor: TfCursor` adds positions, and a `PayloadCursor` adds payloads, as
//!   additive layers (DEC-14). The v1 codec carries no positions, so adding them is a new
//!   parallel section, not a format break.
//! - **Block summaries / upper bounds** — `BlockSummary` reserves `end_doc` + `max_term_freq`
//!   today; richer per-block max-score / block-summary slots can be added behind `BlockCursor`
//!   for Block-Max WAND (Phase 3) without changing `DocCursor`/`TfCursor` callers.
//! - **Borrowed view** — `PostingsView` is a borrowed window over codec bytes plus a
//!   cursor-layer summary slice; the segment block-metadata sidecar lowers into that slice in
//!   ITER-0005 (see `BlockSummary`'s `TODO(ITER-0005)`) with no cursor-API change.

use alloc::vec::Vec;

use crate::codec::{
    BLOCK_DOC_COUNT, BlockDeltaCodec, Codec, CodecError, CodecId, DeltaVarintCodec,
};
use leit_core::{SegmentLocalDocId, TermFreq};

// ============================================================================
// T2: Core Types
// ============================================================================

/// Cursor traversal status.
///
/// Returned by `advance()` and `advance_to()` to signal whether the cursor
/// is positioned at a valid document or has reached the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorStatus {
    /// The cursor is positioned at a valid document.
    Valid,
    /// The cursor has advanced past the end of the postings list.
    Exhausted,
}

/// Decode scratch: workspace-borrowed buffers for cursor decoding.
///
/// The cursor layer reuses this scratch across sequential cursors without reallocating.
/// The buffers are cleared (length reset, capacity retained) between cursors, enabling
/// zero-allocation steady-state traversal (DEC-13, SCENARIO-0039).
///
/// # Ownership model
///
/// The scratch is **workspace-owned** (owned by the query executor) and borrowed by
/// cursors via `&'a mut DecodeScratch`. A cursor borrows it for its lifetime, decodes
/// the codec payload into it, and steps over the decoded values. The scratch is reused
/// across many cursors, so capacity is retained to avoid reallocation.
#[derive(Debug)]
pub struct DecodeScratch {
    /// Decoded document IDs.
    docs: Vec<SegmentLocalDocId>,
    /// Decoded term frequencies (parallel to docs).
    tfs: Vec<TermFreq>,
}

impl DecodeScratch {
    /// Create a new `DecodeScratch` with zero capacity.
    pub const fn new() -> Self {
        Self {
            docs: Vec::new(),
            tfs: Vec::new(),
        }
    }

    /// Create a `DecodeScratch` with pre-allocated capacity.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            docs: Vec::with_capacity(n),
            tfs: Vec::with_capacity(n),
        }
    }

    /// Clear both buffers, retaining their capacity.
    ///
    /// This is called between cursors to reset length while keeping the capacity
    /// for reuse, achieving steady-state zero-allocation (DEC-13).
    pub fn clear(&mut self) {
        self.docs.clear();
        self.tfs.clear();
    }

    /// Get a reference to the decoded documents (internal use).
    pub(crate) fn docs(&self) -> &[SegmentLocalDocId] {
        &self.docs
    }

    /// Get a reference to the decoded term frequencies (internal use).
    pub(crate) fn tfs(&self) -> &[TermFreq] {
        &self.tfs
    }

    /// Get mutable references to both buffers at once.
    ///
    /// Returns both buffers in a single call so a decoder can fill `docs` and `tfs`
    /// together without the borrow checker rejecting two sequential `&mut self` calls.
    pub(crate) fn bufs_mut(&mut self) -> (&mut Vec<SegmentLocalDocId>, &mut Vec<TermFreq>) {
        (&mut self.docs, &mut self.tfs)
    }
}

impl Default for DecodeScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-block summary metadata for WAND / pruning readiness.
///
/// A cursor-layer abstraction of block metadata. Each block carries a document range
/// and a term-frequency upper bound, exposed via `BlockCursor` for Phase 3 pruning.
/// Phase 3 WAND will consume this data without format changes (DEC-04, DEC-06).
///
/// # TODO (ITER-0005)
///
/// The segment block-metadata sidecar (a separate block-summaries table in the segment
/// format) lowers into these cursor-layer summaries. See ITER-0005 STORY-0086.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSummary {
    /// Inclusive end document ID for this block.
    pub end_doc: u32,
    /// Maximum term frequency in this block (upper bound for pruning).
    pub max_term_freq: u32,
}

/// A borrowed view of compressed postings.
///
/// `PostingsView` holds references to the compressed byte payload and the block
/// summaries (if any). It is a fully borrowed type — no ownership, no copy.
/// Multiple cursors can share the same view and step over independent decode
/// positions (SCENARIO-0043).
#[derive(Clone, Copy, Debug)]
pub struct PostingsView<'a> {
    /// Compressed postings bytes (includes 1-byte codec ID prefix).
    bytes: &'a [u8],
    /// Per-block summaries (empty for single-block codecs like `DeltaVarint`).
    block_summaries: &'a [BlockSummary],
}

impl<'a> PostingsView<'a> {
    /// Create a new postings view from bytes and block summaries.
    pub const fn new(bytes: &'a [u8], block_summaries: &'a [BlockSummary]) -> Self {
        Self {
            bytes,
            block_summaries,
        }
    }

    /// Get the compressed byte payload.
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Get the block summaries.
    pub const fn block_summaries(&self) -> &'a [BlockSummary] {
        self.block_summaries
    }
}

/// An upper bound on a single block's maximum score contribution.
///
/// Used by Phase 3 WAND pruning to estimate maximum possible scores.
/// In v1, this is a simple wrapper; Phase 3 may add ranking-specific logic.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct ScoreBound(pub f32);

impl ScoreBound {
    /// Create a new score bound from an f32.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Get the raw f32 value.
    pub const fn as_f32(self) -> f32 {
        self.0
    }
}

// ============================================================================
// T3: Layered Traits + CompressedCursor
// ============================================================================

/// Base cursor trait for document traversal.
///
/// A `DocCursor` allows iteration over document IDs in a postings list.
/// Layered API: `TfCursor: DocCursor` adds term frequency; `BlockCursor: DocCursor`
/// adds block metadata (DEC-14).
pub trait DocCursor {
    /// Get the document ID at the current position, or `None` if exhausted.
    fn current_doc(&self) -> Option<u32>;

    /// Advance to the next document.
    ///
    /// Returns `Valid` if a document is available, `Exhausted` if the cursor
    /// has reached the end.
    fn advance(&mut self) -> CursorStatus;

    /// Seek to a specific document ID or the first document with ID ≥ target.
    ///
    /// **Never moves backward.** If the cursor is already positioned at or past `target`,
    /// no action is taken and the current status is returned.
    ///
    /// Returns `Valid` if a document is found, `Exhausted` if no such document exists.
    fn advance_to(&mut self, target: u32) -> CursorStatus;
}

/// Extended cursor trait with term frequency access.
///
/// A `TfCursor` is a `DocCursor` that also provides term frequency for the
/// current document.
pub trait TfCursor: DocCursor {
    /// Get the term frequency for the current document.
    ///
    /// Behavior is undefined if called when the cursor is exhausted.
    fn current_tf(&self) -> u32;
}

/// Block-aware cursor extension with metadata access.
///
/// A `BlockCursor` is a `DocCursor` that also exposes per-block metadata for
/// pruning readiness (Phase 3). This trait is separate from the base traits to
/// keep simple/hot paths lean and allocation-free (DEC-14).
///
/// **NOTE:** The v1 API exposes the block metadata *surface* only. Pruning/skip
/// *execution* is Phase 3 (STORY-0024 AC-2 deferred). The module keeps executing
/// details internal (pub(crate)) while trait methods are public.
pub trait BlockCursor: DocCursor {
    /// Get the inclusive end document ID of the current block, if available.
    ///
    /// Returns `None` if the cursor is exhausted.
    fn block_end_doc(&self) -> Option<u32>;

    /// Get an upper bound on the maximum score contribution from this block.
    ///
    /// Used by Phase 3 WAND pruning to estimate score bounds.
    /// Returns a score bound suitable for comparison against early-exit thresholds.
    fn block_max_score(&self) -> ScoreBound;
}

/// Compressed postings cursor — the canonical compressed-postings cursor implementation.
///
/// Decodes a `PostingsView` (encoded via `DeltaVarint` or `BlockDelta`) once into a
/// borrowed `DecodeScratch` and traverses the decoded slices, using binary search for
/// `advance_to`. Users typically construct it via `CursorFactory::open_doc_cursor` rather
/// than directly; the type is public so the trait impls are nameable and for explicit use
/// in high-performance code paths.
///
/// # Lifetime
///
/// The cursor borrows the scratch for the duration of traversal. After a cursor is
/// dropped, the scratch can be cleared and reused for the next cursor.
#[derive(Debug)]
pub struct CompressedCursor<'a> {
    /// Decoded doc IDs (borrowed from scratch).
    docs: &'a [SegmentLocalDocId],
    /// Decoded term frequencies (borrowed from scratch).
    tfs: &'a [TermFreq],
    /// Block summaries (borrowed from view).
    block_summaries: &'a [BlockSummary],
    /// Current position in the decoded docs array.
    pos: usize,
}

impl<'a> CompressedCursor<'a> {
    /// Create a new cursor from a view and scratch.
    ///
    /// Decodes the view's payload into the scratch, clears the scratch first,
    /// and positions at the start.
    fn new(view: PostingsView<'a>, scratch: &'a mut DecodeScratch) -> Result<Self, CodecError> {
        scratch.clear();
        let bytes = view.bytes();
        let block_summaries = view.block_summaries();

        // Decode using the same logic as decode_any, but with split borrows via bufs_mut().
        if bytes.is_empty() {
            return Err(CodecError::Truncated);
        }
        let marker = bytes[0];
        let (out_docs, out_tfs) = scratch.bufs_mut();
        match CodecId::from_u8(marker) {
            Some(CodecId::DeltaVarint) => {
                DeltaVarintCodec.decode(bytes, out_docs, out_tfs)?;
            }
            Some(CodecId::BlockDelta) => {
                BlockDeltaCodec.decode(bytes, out_docs, out_tfs)?;
            }
            None => return Err(CodecError::BadMarker(marker)),
        }

        Ok(Self {
            docs: scratch.docs(),
            tfs: scratch.tfs(),
            block_summaries,
            pos: 0,
        })
    }

    /// Get the block index for a given position (which 128-doc block).
    fn block_index_for_pos(&self, pos: usize) -> usize {
        pos / BLOCK_DOC_COUNT
    }

    /// Get the block summary for the current position.
    fn current_block_summary(&self) -> Option<BlockSummary> {
        if self.pos >= self.docs.len() {
            return None;
        }
        let block_idx = self.block_index_for_pos(self.pos);
        self.block_summaries.get(block_idx).copied()
    }
}

impl<'a> DocCursor for CompressedCursor<'a> {
    fn current_doc(&self) -> Option<u32> {
        self.docs.get(self.pos).map(|d| d.get())
    }

    fn advance(&mut self) -> CursorStatus {
        if self.pos < self.docs.len() {
            self.pos += 1;
        }
        if self.pos < self.docs.len() {
            CursorStatus::Valid
        } else {
            CursorStatus::Exhausted
        }
    }

    fn advance_to(&mut self, target: u32) -> CursorStatus {
        // Binary search over remaining docs to find the first doc >= target.
        if self.pos >= self.docs.len() {
            return CursorStatus::Exhausted;
        }
        let remaining = &self.docs[self.pos..];
        match remaining.binary_search_by_key(&target, |d| d.get()) {
            Ok(offset) => {
                // Exact match.
                self.pos += offset;
                CursorStatus::Valid
            }
            Err(offset) => {
                // Insert position: first doc > target.
                self.pos += offset;
                if self.pos < self.docs.len() {
                    CursorStatus::Valid
                } else {
                    CursorStatus::Exhausted
                }
            }
        }
    }
}

impl<'a> TfCursor for CompressedCursor<'a> {
    fn current_tf(&self) -> u32 {
        self.tfs.get(self.pos).map(|t| t.get()).unwrap_or(0)
    }
}

impl<'a> BlockCursor for CompressedCursor<'a> {
    fn block_end_doc(&self) -> Option<u32> {
        self.current_block_summary().map(|s| s.end_doc)
    }

    fn block_max_score(&self) -> ScoreBound {
        self.current_block_summary()
            .map(|s| ScoreBound::new(s.max_term_freq as f32))
            .unwrap_or(ScoreBound::new(0.0))
    }
}

// ============================================================================
// T4: CursorFactory
// ============================================================================

/// Factory for creating cursors from postings views.
///
/// Trait over concrete cursor implementations, enabling codec-agnostic factories
/// that dispatch on the 1-byte codec marker (SCENARIO-0041).
pub trait CursorFactory<'a> {
    /// The concrete cursor type returned by this factory.
    type Cursor: DocCursor + 'a;

    /// Open a document cursor from a postings view using a borrowed scratch.
    ///
    /// The factory reads the codec ID marker and decodes the payload, then
    /// returns a positioned cursor. Errors if the codec ID is unrecognized
    /// or decoding fails.
    fn open_doc_cursor(
        &self,
        view: PostingsView<'a>,
        scratch: &'a mut DecodeScratch,
    ) -> Result<Self::Cursor, CodecError>;
}

/// A codec-agnostic cursor factory.
///
/// Reads the codec ID from the postings bytes and creates a `CompressedCursor`.
/// Works with both `DeltaVarint` and `BlockDelta` encodings (SCENARIO-0041).
#[derive(Debug)]
pub struct DefaultCursorFactory;

impl<'a> CursorFactory<'a> for DefaultCursorFactory {
    type Cursor = CompressedCursor<'a>;

    fn open_doc_cursor(
        &self,
        view: PostingsView<'a>,
        scratch: &'a mut DecodeScratch,
    ) -> Result<Self::Cursor, CodecError> {
        CompressedCursor::new(view, scratch)
    }
}

// ============================================================================
// T5: BlockDecoder
// ============================================================================

/// Decode a single block independently.
///
/// The `BlockDecoder` trait decodes an individual block within a multi-block
/// encoding (e.g., `BlockDelta`) without decoding the entire postings list.
/// This enables selective block decoding for Phase 3 skip-list optimization
/// (deferred: STORY-0024 AC-3).
///
/// In v1, this is defined but not heavily used; Phase 3 will invoke it for
/// block-selective decode in conjunction with `BlockCursor` metadata.
pub trait BlockDecoder {
    /// Decode a specific block by ordinal from an encoded postings stream.
    ///
    /// Writes the decoded doc IDs and term frequencies into caller-provided buffers,
    /// which are **cleared and refilled** (consistent with `Codec::decode`) so they
    /// can be reused across calls (typically a `DecodeScratch`'s buffers).
    ///
    /// # Arguments
    ///
    /// - `bytes`: the full encoded postings stream (with the 1-byte codec marker).
    /// - `block_id`: ordinal of the block to decode (0-indexed).
    /// - `out_docs`: output buffer for doc IDs (cleared, then filled).
    /// - `out_tfs`: output buffer for term frequencies (cleared, then filled).
    ///
    /// # Returns
    ///
    /// The count of documents decoded in the block, `Ok(0)` if `block_id` is past the
    /// last block, or a `CodecError` on malformed input.
    fn decode_block(
        &self,
        bytes: &[u8],
        block_id: usize,
        out_docs: &mut Vec<SegmentLocalDocId>,
        out_tfs: &mut Vec<TermFreq>,
    ) -> Result<usize, CodecError>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{BlockDeltaCodec, Codec, DeltaVarintCodec};
    use alloc::boxed::Box;
    use alloc::vec;

    /// SCENARIO-0043: `PostingsView` borrows bytes and block summaries (no copy).
    /// Multiple cursors can share the same view with independent positions.
    #[test]
    fn postings_view_borrows() {
        let bytes = [1_u8, 2_u8, 3_u8];
        let summaries = [BlockSummary {
            end_doc: 100,
            max_term_freq: 5,
        }];

        let view = PostingsView::new(&bytes, &summaries);

        // View holds borrowed references, not owned data.
        assert_eq!(view.bytes(), &[1_u8, 2_u8, 3_u8]);
        assert_eq!(view.block_summaries().len(), 1);
        assert_eq!(view.block_summaries()[0].end_doc, 100);
    }

    /// SCENARIO-0039: `DecodeScratch` `clear()` retains capacity (no realloc).
    #[test]
    fn scratch_reuse_retains_capacity() {
        let mut scratch = DecodeScratch::with_capacity(100);
        let (docs, tfs) = scratch.bufs_mut();
        docs.extend([SegmentLocalDocId::new(10), SegmentLocalDocId::new(20)]);
        tfs.extend([TermFreq::new(1), TermFreq::new(2)]);

        let initial_capacity = scratch.docs.capacity();
        assert!(initial_capacity >= 100);

        scratch.clear();
        assert_eq!(scratch.docs().len(), 0);
        assert_eq!(scratch.tfs().len(), 0);
        assert_eq!(scratch.docs.capacity(), initial_capacity);
    }

    /// SCENARIO-0034: Roundtrip encode → decode → traverse.
    /// Encode [(10,3), (25,7), (100,2)], traverse yields [10,25,100] and [3,7,2].
    #[test]
    fn roundtrip() {
        let codec = DeltaVarintCodec;
        let postings = [
            (SegmentLocalDocId::new(10), TermFreq::new(3)),
            (SegmentLocalDocId::new(25), TermFreq::new(7)),
            (SegmentLocalDocId::new(100), TermFreq::new(2)),
        ];

        let encoded = codec.encode(&postings);
        let view = PostingsView::new(&encoded, &[]);

        let mut scratch = DecodeScratch::new();
        let mut cursor = DefaultCursorFactory
            .open_doc_cursor(view, &mut scratch)
            .expect("decode should succeed");

        // Traverse and collect results.
        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        while let Some(doc) = cursor.current_doc() {
            docs.push(doc);
            tfs.push(cursor.current_tf());
            if cursor.advance() == CursorStatus::Exhausted {
                break;
            }
        }

        assert_eq!(docs, vec![10, 25, 100]);
        assert_eq!(tfs, vec![3, 7, 2]);
    }

    /// SCENARIO-0035: `advance()` returns Valid/Exhausted in order, no skips/dups.
    #[test]
    fn advance_status() {
        let codec = DeltaVarintCodec;
        let postings = [
            (SegmentLocalDocId::new(5), TermFreq::new(1)),
            (SegmentLocalDocId::new(12), TermFreq::new(2)),
            (SegmentLocalDocId::new(30), TermFreq::new(3)),
        ];

        let encoded = codec.encode(&postings);
        let view = PostingsView::new(&encoded, &[]);

        let mut scratch = DecodeScratch::new();
        let mut cursor = DefaultCursorFactory
            .open_doc_cursor(view, &mut scratch)
            .expect("decode should succeed");

        // Start: at 5.
        assert_eq!(cursor.current_doc(), Some(5));
        assert_eq!(cursor.advance(), CursorStatus::Valid);

        // At 12.
        assert_eq!(cursor.current_doc(), Some(12));
        assert_eq!(cursor.advance(), CursorStatus::Valid);

        // At 30.
        assert_eq!(cursor.current_doc(), Some(30));
        assert_eq!(cursor.advance(), CursorStatus::Exhausted);

        // Exhausted.
        assert_eq!(cursor.current_doc(), None);
    }

    /// SCENARIO-0036: `advance_to()` never moves backward; seek semantics.
    /// Over [5, 12, 30, 100]: seek(25)→30, seek(12)→stay 30, seek(200)→Exhausted.
    #[test]
    fn advance_to() {
        let codec = DeltaVarintCodec;
        let postings = [
            (SegmentLocalDocId::new(5), TermFreq::new(1)),
            (SegmentLocalDocId::new(12), TermFreq::new(2)),
            (SegmentLocalDocId::new(30), TermFreq::new(3)),
            (SegmentLocalDocId::new(100), TermFreq::new(4)),
        ];

        let encoded = codec.encode(&postings);
        let view = PostingsView::new(&encoded, &[]);

        let mut scratch = DecodeScratch::new();
        let mut cursor = DefaultCursorFactory
            .open_doc_cursor(view, &mut scratch)
            .expect("decode should succeed");

        // Start at 5.
        assert_eq!(cursor.current_doc(), Some(5));

        // Seek to 25 → lands on 30.
        assert_eq!(cursor.advance_to(25), CursorStatus::Valid);
        assert_eq!(cursor.current_doc(), Some(30));

        // Seek to 12 (backward) → stays at 30.
        assert_eq!(cursor.advance_to(12), CursorStatus::Valid);
        assert_eq!(cursor.current_doc(), Some(30));

        // Seek to 100 → lands on 100.
        assert_eq!(cursor.advance_to(100), CursorStatus::Valid);
        assert_eq!(cursor.current_doc(), Some(100));

        // Seek to 200 → past end.
        assert_eq!(cursor.advance_to(200), CursorStatus::Exhausted);
        assert_eq!(cursor.current_doc(), None);
    }

    /// SCENARIO-0041: Both `DeltaVarint` and `BlockDelta` via same factory yield identical output.
    #[test]
    fn factory_codec_interchangeable() {
        let postings = [
            (SegmentLocalDocId::new(5), TermFreq::new(1)),
            (SegmentLocalDocId::new(15), TermFreq::new(2)),
            (SegmentLocalDocId::new(25), TermFreq::new(3)),
        ];

        let factory = DefaultCursorFactory;

        // Encode with DeltaVarint.
        let delta_encoded = DeltaVarintCodec.encode(&postings);
        let delta_view = PostingsView::new(&delta_encoded, &[]);

        let mut scratch1 = DecodeScratch::new();
        let mut cursor1 = factory
            .open_doc_cursor(delta_view, &mut scratch1)
            .expect("DeltaVarint should decode");

        let mut delta_docs = Vec::new();
        let mut delta_tfs = Vec::new();
        while let Some(doc) = cursor1.current_doc() {
            delta_docs.push(doc);
            delta_tfs.push(cursor1.current_tf());
            if cursor1.advance() == CursorStatus::Exhausted {
                break;
            }
        }

        // Encode with BlockDelta.
        let block_encoded = BlockDeltaCodec.encode(&postings);
        let block_view = PostingsView::new(&block_encoded, &[]);

        let mut scratch2 = DecodeScratch::new();
        let mut cursor2 = factory
            .open_doc_cursor(block_view, &mut scratch2)
            .expect("BlockDelta should decode");

        let mut block_docs = Vec::new();
        let mut block_tfs = Vec::new();
        while let Some(doc) = cursor2.current_doc() {
            block_docs.push(doc);
            block_tfs.push(cursor2.current_tf());
            if cursor2.advance() == CursorStatus::Exhausted {
                break;
            }
        }

        // Both should yield identical results.
        assert_eq!(delta_docs, block_docs);
        assert_eq!(delta_tfs, block_tfs);
    }

    /// SCENARIO-0042: `BlockDecoder::decode_block` decodes one block independently of any
    /// cursor, returns the decoded count, and refills the same reused buffers for a different
    /// block id. Uses a 200-doc list so `BlockDelta` (128-doc blocks) produces two blocks.
    #[test]
    fn block_decoder_independent() {
        // Trait is object-safe (usable behind a trait object).
        fn _trait_is_object_safe(_: &dyn BlockDecoder) {}

        // Build a 200-posting list: doc i -> (i*2, i+1). Two BlockDelta blocks (128 + 72).
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..200_u32)
            .map(|i| (SegmentLocalDocId::new(i * 2), TermFreq::new(i + 1)))
            .collect();
        let encoded = BlockDeltaCodec.encode(&postings);

        let mut docs = Vec::new();
        let mut tfs = Vec::new();

        // Block 0: first 128 docs.
        let count0 = BlockDeltaCodec
            .decode_block(&encoded, 0, &mut docs, &mut tfs)
            .expect("block 0 decodes");
        assert_eq!(count0, 128);
        assert_eq!(docs.first().map(|d| d.get()), Some(0));
        assert_eq!(docs.last().map(|d| d.get()), Some(127 * 2));
        assert_eq!(tfs[0].get(), 1);

        // Block 1: remaining 72 docs, decoded into the SAME reused buffers (cleared+refilled).
        let count1 = BlockDeltaCodec
            .decode_block(&encoded, 1, &mut docs, &mut tfs)
            .expect("block 1 decodes");
        assert_eq!(count1, 72);
        assert_eq!(docs.len(), 72);
        assert_eq!(docs.first().map(|d| d.get()), Some(128 * 2));
        assert_eq!(docs.last().map(|d| d.get()), Some(199 * 2));
        assert_eq!(tfs.last().map(|t| t.get()), Some(200));

        // Out-of-range block id yields an empty decode.
        let count_oob = BlockDeltaCodec
            .decode_block(&encoded, 2, &mut docs, &mut tfs)
            .expect("oob block id is Ok(0)");
        assert_eq!(count_oob, 0);
        assert!(docs.is_empty());
    }

    /// SCENARIO-0038: `BlockCursor` exposes block `end_doc` and `max_score` per block.
    #[test]
    fn block_cursor_boundaries() {
        let codec = DeltaVarintCodec;
        let postings = [
            (SegmentLocalDocId::new(10), TermFreq::new(3)),
            (SegmentLocalDocId::new(20), TermFreq::new(7)),
        ];

        let encoded = codec.encode(&postings);

        // For a single-block (DeltaVarint) list, create a block summary.
        let summaries = [BlockSummary {
            end_doc: 20,
            max_term_freq: 7,
        }];

        let view = PostingsView::new(&encoded, &summaries);

        let mut scratch = DecodeScratch::new();
        let cursor: Box<dyn BlockCursor> =
            Box::new(CompressedCursor::new(view, &mut scratch).expect("decode should succeed"));

        // At the first block.
        assert_eq!(cursor.block_end_doc(), Some(20));
        assert_eq!(cursor.block_max_score().as_f32(), 7.0);
    }

    /// SCENARIO-0002 + SCENARIO-0009: `BlockCursor` trait is public; exposes API surface only.
    /// No pruning/skip execution (deferred Phase 3).
    #[test]
    fn block_api_contract() {
        // Verify BlockCursor trait is public and accessible.
        fn _trait_is_object_safe(_: &dyn BlockCursor) {}

        // Verify the trait methods are queryable.
        let codec = DeltaVarintCodec;
        let postings = [(SegmentLocalDocId::new(50), TermFreq::new(5))];

        let encoded = codec.encode(&postings);
        let summaries = [BlockSummary {
            end_doc: 50,
            max_term_freq: 5,
        }];

        let view = PostingsView::new(&encoded, &summaries);

        let mut scratch = DecodeScratch::new();
        let cursor = CompressedCursor::new(view, &mut scratch).expect("decode should succeed");

        // API is public and callable.
        assert_eq!(cursor.block_end_doc(), Some(50));
        let bound = cursor.block_max_score();
        assert!(bound.as_f32() >= 5.0);
    }

    /// SCENARIO-0002/0009 (edge): for a single-block codec with NO block summaries,
    /// the block-aware API returns a conservative default (`None` / `ScoreBound(0.0)`)
    /// rather than fabricating a bound. Documents the empty-`block_summaries` contract.
    #[test]
    fn block_api_empty_summaries_conservative_default() {
        let encoded = DeltaVarintCodec.encode(&[
            (SegmentLocalDocId::new(5), TermFreq::new(2)),
            (SegmentLocalDocId::new(9), TermFreq::new(4)),
        ]);
        // No summaries supplied (the common single-block / DeltaVarint case).
        let view = PostingsView::new(&encoded, &[]);
        let mut scratch = DecodeScratch::new();
        let cursor = CompressedCursor::new(view, &mut scratch).expect("decode should succeed");

        // Positioned at a valid doc, but without summaries the block API is conservative.
        assert_eq!(cursor.current_doc(), Some(5));
        assert_eq!(cursor.block_end_doc(), None);
        assert_eq!(cursor.block_max_score().as_f32(), 0.0);
    }

    /// SCENARIO-0039 + SCENARIO-0019/0024: one `DecodeScratch` reused across N sequential
    /// cursors over different postings lists allocates no new buffer capacity after warm-up
    /// (DEC-13 workspace-borrowed ownership). Capacity-stability is the no_std-friendly proxy
    /// for "zero allocations across cursor lifetimes".
    #[test]
    fn scratch_reuse_no_realloc() {
        // Pre-size the scratch so the first cursor does not need to grow it.
        let mut scratch = DecodeScratch::with_capacity(64);

        let lists: [&[(SegmentLocalDocId, TermFreq)]; 3] = [
            &[
                (SegmentLocalDocId::new(1), TermFreq::new(1)),
                (SegmentLocalDocId::new(4), TermFreq::new(2)),
            ],
            &[
                (SegmentLocalDocId::new(2), TermFreq::new(3)),
                (SegmentLocalDocId::new(8), TermFreq::new(1)),
                (SegmentLocalDocId::new(9), TermFreq::new(5)),
            ],
            &[(SegmentLocalDocId::new(7), TermFreq::new(9))],
        ];

        let mut cap_after_warmup = None;
        for list in lists {
            let encoded = DeltaVarintCodec.encode(list);
            let view = PostingsView::new(&encoded, &[]);
            {
                let mut cursor = DefaultCursorFactory
                    .open_doc_cursor(view, &mut scratch)
                    .expect("decode should succeed");
                // Drain the cursor (hot path) before the borrow ends.
                while cursor.current_doc().is_some() {
                    let _ = cursor.current_tf();
                    if cursor.advance() == CursorStatus::Exhausted {
                        break;
                    }
                }
            }
            // After the first cursor, capacity must never grow again.
            let cap = scratch.docs.capacity();
            match cap_after_warmup {
                None => cap_after_warmup = Some(cap),
                Some(prev) => assert_eq!(cap, prev, "scratch reallocated across cursors"),
            }
        }
    }

    /// SCENARIO-0040: hot-path traversal (`advance`, `current_doc`, `advance_to`) over an
    /// already-decoded cursor causes no buffer growth, and the cursor struct stays small.
    ///
    /// Decoding happens once at cursor construction; the hot-path methods only step an
    /// index over the borrowed slices. With a pre-sized scratch large enough to hold the
    /// decoded list, capacity is unchanged after a long traversal loop.
    #[test]
    fn hot_path_capacity_stable_and_cursor_small() {
        // Cursor is a lightweight, borrow-only struct (3 slices + an index).
        assert!(
            size_of::<CompressedCursor<'_>>() < 100,
            "CompressedCursor should be a small stack value"
        );

        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..50_u32)
            .map(|i| (SegmentLocalDocId::new(i * 3), TermFreq::new(i + 1)))
            .collect();
        let encoded = BlockDeltaCodec.encode(&postings);
        let view = PostingsView::new(&encoded, &[]);
        // Pre-size large enough that decoding 50 docs never grows the buffers.
        let mut scratch = DecodeScratch::with_capacity(64);

        {
            let mut cursor = DefaultCursorFactory
                .open_doc_cursor(view, &mut scratch)
                .expect("decode should succeed");
            // Tight hot-path loop: advance + read, plus a seek when exhausted.
            for _ in 0..1000 {
                let _ = cursor.current_doc();
                let _ = cursor.current_tf();
                if cursor.advance() == CursorStatus::Exhausted {
                    // Already exhausted: advance_to must stay exhausted (no backward move).
                    assert_eq!(cursor.advance_to(0), CursorStatus::Exhausted);
                    break;
                }
            }
        }
        // Decoding 50 docs into a 64-capacity scratch must not have reallocated.
        assert_eq!(scratch.docs.capacity(), 64);
    }
}
