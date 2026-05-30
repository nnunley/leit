// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Block metadata section schema (v1).
//!
//! A per-block summary sidecar for postings blocks. It lets block-aware queries (max-score
//! pruning, doc-range filtering) skip blocks without decoding postings data. The section holds
//! one fixed-width [`BlockMetadataEntry`] per postings block, with the blocks of all terms stored
//! contiguously in a single table.
//!
//! **Physical layout:**
//! - A single contiguous table of [`BlockMetadataEntry`] (12 bytes each), addressed by the
//!   `block_meta` section offset in the segment header.
//! - A term locates its own blocks through a first-block index and block count carried on its
//!   postings-table entry (written by the segment writer).
//! - Each entry holds the block's inclusive end document, a scorer-agnostic impact upper bound,
//!   and a relative offset to the block's compressed payload.
//!
//! **Doc-range is implicit and scoped per term.** Only `end_doc` is stored. Within one term's
//! contiguous block range, a block's lower bound is the previous block's `end_doc + 1`. This rule
//! must NOT cross a term boundary: a term's first block has no table-derivable lower bound (it is
//! the term's first posting). Skipping needs only the monotonic per-block `end_doc` upper bounds,
//! so the first block's exact lower bound is deliberately not stored. The final block of a term may
//! be shorter than a full block; its `end_doc` is the term's true last document, not padded.

use bytemuck::{Pod, Zeroable};

/// Per-block summary entry in the block-metadata sidecar.
///
/// Each entry is **12 bytes, little-endian, zero-copy POD**. A segment's postings blocks are stored
/// contiguously in the block-metadata section; a term locates its block summaries through the
/// first-block index and block count on its postings-table entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct BlockMetadataEntry {
    /// Inclusive end document ID for this block (u32 LE). The lower bound is implicit and re-derived
    /// per term (see the module docs); it is never derived across a term boundary.
    pub end_doc: u32,
    /// Maximum term frequency in this block (u32 LE): a scorer-agnostic impact upper bound, not a
    /// scored value. A pruning executor derives a score bound from this at query time, so no scorer
    /// parameters live in the segment.
    pub max_term_freq: u32,
    /// Byte offset to this block's compressed payload, **relative** to the term's postings payload
    /// start (u32 LE) — never an absolute file offset, so merge can recompute it from re-encoded
    /// postings without changing the format.
    pub decode_offset: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn block_metadata_entry_layout() {
        // Verify byte layout: 12 bytes total, fields in order (end_doc, max_term_freq, decode_offset).
        // Alignment is 4 bytes (natural alignment of u32 in #[repr(C)]).
        assert_eq!(size_of::<BlockMetadataEntry>(), 12);
        assert_eq!(align_of::<BlockMetadataEntry>(), 4);
    }

    #[test]
    fn block_metadata_entry_round_trip() {
        // Create an entry with known values.
        let original = BlockMetadataEntry {
            end_doc: 127,
            max_term_freq: 42,
            decode_offset: 1024,
        };

        // Convert to bytes via bytemuck cast.
        let bytes: [u8; 12] = bytemuck::cast(original);

        // Verify little-endian encoding: end_doc=127 (0x7F) at offset 0.
        assert_eq!(bytes[0..4], 127_u32.to_le_bytes());
        // max_term_freq=42 (0x2A) at offset 4.
        assert_eq!(bytes[4..8], 42_u32.to_le_bytes());
        // decode_offset=1024 (0x0400) at offset 8.
        assert_eq!(bytes[8..12], 1024_u32.to_le_bytes());

        // Convert back and verify identity.
        let restored: BlockMetadataEntry = bytemuck::cast(bytes);
        assert_eq!(restored, original);
    }

    #[test]
    fn block_metadata_entry_multiple_round_trip() {
        // Test a sequence of entries to verify contiguous-array handling.
        let entries = [
            BlockMetadataEntry {
                end_doc: 127,
                max_term_freq: 50,
                decode_offset: 0,
            },
            BlockMetadataEntry {
                end_doc: 255,
                max_term_freq: 30,
                decode_offset: 512,
            },
            BlockMetadataEntry {
                end_doc: 300,
                max_term_freq: 10,
                decode_offset: 1024,
            },
        ];

        // Cast the entire array to bytes.
        let bytes: &[u8] = bytemuck::cast_slice(&entries);

        // Verify length: 3 entries × 12 bytes = 36 bytes.
        assert_eq!(bytes.len(), 36);

        // Cast back and verify identity.
        let restored: &[BlockMetadataEntry] = bytemuck::cast_slice(bytes);
        assert_eq!(restored, entries);
    }

    #[test]
    fn block_metadata_entry_zero_init() {
        // Verify that zero-initialization (via Zeroable) produces a valid all-zeros entry.
        let zeroed: BlockMetadataEntry = Zeroable::zeroed();
        let bytes = bytemuck::bytes_of(&zeroed);
        assert_eq!(bytes, &[0_u8; 12]);
    }
}
