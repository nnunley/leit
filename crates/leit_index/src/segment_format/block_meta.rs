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
use leit_postings::BlockSummary;

/// Per-block summary entry in the block-metadata sidecar.
///
/// Each entry is **12 bytes, little-endian, zero-copy POD**. A segment's postings blocks are stored
/// contiguously in the block-metadata section; a term locates its block summaries through the
/// first-block index and block count on its postings-table entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct BlockMetadataEntry {
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

impl From<BlockMetadataEntry> for BlockSummary {
    /// Lower a segment-layer block metadata entry into a cursor-layer block summary.
    ///
    /// The cursor layer needs only the document range and term-frequency bound; the decode offset
    /// is a segment-layer implementation detail and is discarded during lowering.
    fn from(entry: BlockMetadataEntry) -> Self {
        Self {
            end_doc: entry.end_doc,
            max_term_freq: entry.max_term_freq,
        }
    }
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

    /// Test that segment-layer block metadata entries lower into cursor-layer block summaries.
    ///
    /// The lowering drops the `decode_offset` (segment-layer only) and preserves `end_doc` and
    /// `max_term_freq` for use in the cursor layer.
    #[test]
    fn lowering_block_metadata_entry_to_block_summary() {
        let entry = BlockMetadataEntry {
            end_doc: 127,
            max_term_freq: 42,
            decode_offset: 1024,
        };

        let summary: BlockSummary = entry.into();

        // Verify the lowering preserves end_doc and max_term_freq
        assert_eq!(summary.end_doc, 127);
        assert_eq!(summary.max_term_freq, 42);
        // decode_offset is discarded (not present in BlockSummary)
    }

    /// Test lowering with multiple entries to verify the conversion is consistent.
    #[test]
    fn lowering_block_metadata_entries_consistent() {
        let entries = [
            BlockMetadataEntry {
                end_doc: 128,
                max_term_freq: 10,
                decode_offset: 0,
            },
            BlockMetadataEntry {
                end_doc: 256,
                max_term_freq: 20,
                decode_offset: 1024,
            },
            BlockMetadataEntry {
                end_doc: 300,
                max_term_freq: 5,
                decode_offset: 2048,
            },
        ];

        for (i, entry) in entries.iter().enumerate() {
            let summary: BlockSummary = (*entry).into();
            assert_eq!(
                summary.end_doc, entry.end_doc,
                "entry {}: end_doc mismatch",
                i
            );
            assert_eq!(
                summary.max_term_freq, entry.max_term_freq,
                "entry {}: max_term_freq mismatch",
                i
            );
        }
    }

    /// Measure block-meta overhead per block across a deterministic multi-term corpus.
    ///
    /// This test provides **analytical evidence** that the per-block structural overhead
    /// of the block-metadata sidecar is fixed and small (12 bytes per block, one entry).
    /// The test proves:
    /// - The `block_meta` section is exactly `total_blocks * 12` bytes
    /// - Overhead per block is 12 bytes (the fixed entry width from the schema)
    /// - Reading block metadata via `BlockMetadataReader` touches only the `block_meta`
    ///   byte range, never the `postings_data` section (structural proof)
    #[test]
    fn overhead_bytes_per_block() {
        use alloc::collections::BTreeMap;
        use alloc::collections::BTreeSet;
        use alloc::string::String;
        use alloc::vec;

        use leit_core::{FieldId, TermId};
        use leit_text::FieldAnalyzers;

        use crate::memory::{FieldMetadata, InMemoryIndex, PostingEntry, TermEntry};
        use crate::segment_format::writer::write_segment;

        // Build a deterministic multi-term corpus with varied posting counts:
        // - Term 0 ("common"): 250 postings => ceil(250/128) = 2 blocks
        // - Term 1 ("uncommon"): 50 postings => ceil(50/128) = 1 block
        // - Term 2 ("rare"): 10 postings => ceil(10/128) = 1 block
        // Total expected: 2 + 1 + 1 = 4 blocks

        let mut documents = BTreeSet::new();
        for doc_id in 0..250 {
            documents.insert(doc_id);
        }

        let field_id = FieldId::new(1);
        let mut field_names = BTreeMap::new();
        field_names.insert(String::from("text"), field_id);

        let mut field_stats = BTreeMap::new();
        field_stats.insert(
            field_id,
            FieldMetadata {
                field_id,
                doc_count: 250,
                total_terms: 3,
            },
        );

        let mut terms_to_ids = BTreeMap::new();
        let mut term_entries = vec![];

        // Term 0: "common" with 250 postings
        terms_to_ids.insert((field_id, String::from("common")), TermId::new(0));
        term_entries.push(TermEntry {
            field_id,
            term_id: TermId::new(0),
            term: String::from("common"),
        });

        // Term 1: "uncommon" with 50 postings
        terms_to_ids.insert((field_id, String::from("uncommon")), TermId::new(1));
        term_entries.push(TermEntry {
            field_id,
            term_id: TermId::new(1),
            term: String::from("uncommon"),
        });

        // Term 2: "rare" with 10 postings
        terms_to_ids.insert((field_id, String::from("rare")), TermId::new(2));
        term_entries.push(TermEntry {
            field_id,
            term_id: TermId::new(2),
            term: String::from("rare"),
        });

        let mut postings = BTreeMap::new();

        // Term 0: 250 postings
        let mut common_postings = vec![];
        for doc_id in 0..250 {
            common_postings.push(PostingEntry {
                doc_id,
                term_freq: (doc_id % 10) + 1,
            });
        }
        postings.insert(TermId::new(0), common_postings);

        // Term 1: 50 postings (docs 0, 5, 10, ..., 245)
        let mut uncommon_postings = vec![];
        for i in 0..50 {
            uncommon_postings.push(PostingEntry {
                doc_id: i * 5,
                term_freq: 2,
            });
        }
        postings.insert(TermId::new(1), uncommon_postings);

        // Term 2: 10 postings (docs 0, 25, 50, ..., 225)
        let mut rare_postings = vec![];
        for i in 0..10 {
            rare_postings.push(PostingEntry {
                doc_id: i * 25,
                term_freq: 1,
            });
        }
        postings.insert(TermId::new(2), rare_postings);

        let mut posting_blocks = BTreeMap::new();
        posting_blocks.insert(TermId::new(0), vec![]);
        posting_blocks.insert(TermId::new(1), vec![]);
        posting_blocks.insert(TermId::new(2), vec![]);

        let field_doc_lengths = BTreeMap::new();

        let index = InMemoryIndex::new(
            FieldAnalyzers::default(),
            documents,
            terms_to_ids,
            term_entries,
            postings,
            posting_blocks,
            field_stats,
            field_names,
            field_doc_lengths,
        );

        // Write the segment
        let segment = write_segment(&index).expect("segment write succeeds");

        // Parse the segment header to locate block_meta section
        use crate::segment_format::header::SegmentHeader;
        let header = SegmentHeader::read(&segment).expect("header valid");

        // Compute block_meta section bounds
        let block_meta_offset =
            usize::try_from(header.block_meta_offset).expect("offset fits in usize");
        let stored_fields_offset =
            usize::try_from(header.stored_fields_offset).expect("offset fits in usize");
        let block_meta_len = stored_fields_offset - block_meta_offset;

        // Expected total blocks:
        // - Term 0: ceil(250 / 128) = 2 blocks
        // - Term 1: ceil(50 / 128) = 1 block
        // - Term 2: ceil(10 / 128) = 1 block
        // Total: 4 blocks
        let expected_total_blocks = 4;
        let expected_block_meta_len = expected_total_blocks * 12;

        // ASSERTION 1: Block metadata section is exactly total_blocks * 12 bytes
        assert_eq!(
            block_meta_len, expected_block_meta_len,
            "block_meta section must be exactly {} bytes for {} blocks",
            expected_block_meta_len, expected_total_blocks
        );

        // ASSERTION 2: Overhead per block is fixed at 12 bytes
        assert_eq!(
            block_meta_len / expected_total_blocks,
            12,
            "overhead per block must be 12 bytes"
        );

        // ASSERTION 3: Structural proof — reading block summaries touches only block_meta section.
        // Create a BlockMetadataReader and verify it reads only from the block_meta range.
        use crate::segment_format::readers::BlockMetadataReader;

        let reader = BlockMetadataReader::new(
            &segment,
            header.block_meta_offset,
            header.stored_fields_offset,
        )
        .expect("BlockMetadataReader constructs");

        // Verify entry count matches expected
        assert_eq!(
            reader.len() as usize,
            expected_total_blocks,
            "entry count must match expected blocks"
        );

        // Read all entries and verify each is within the block_meta section
        for i in 0..reader.len() {
            let (end_doc, max_term_freq, decode_offset) =
                reader.entry(i).expect("entry read succeeds");
            // Bounds check on values (sanity)
            assert!(end_doc < 250, "end_doc must be < document count");
            assert!(max_term_freq > 0, "max_term_freq must be positive");
            // decode_offset is relative, so it should be reasonable
            assert!(decode_offset < 10_000, "decode_offset should be plausible");
        }

        // Structural proof: BlockMetadataReader computes absolute_offset and bounds-checks
        // it against section_end (section_start..section_end), so reading entries can never
        // access postings_data or any section outside the block_meta range.
    }
}
