// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Borrowed segment view: fully-borrowed, zero-copy access to segment sections.
//!
//! `SegmentView<'a>` validates a segment buffer and provides lifetime-safe accessors to each
//! section (field table, lexicon, postings table, postings data, block metadata) as borrowed readers.
//! Three validation modes (`HeaderOnly`, `Structural`, `Full`) provide a tradeoff between startup latency
//! and error detection.
//!
//! The view is the canonical public API for the current format; the legacy directory reader
//! is deprecated and frozen as `DirectorySegmentView`.

use crate::error::{SegmentError, ValidationMode};

use super::footer;
use super::header::SegmentHeader;
use super::readers::{
    BlockMetadataReader, FieldTableReader, LexiconReader, PostingsDataReader, PostingsTableReader,
};

/// A fully-borrowed, zero-copy view of a segment buffer.
///
/// Holds a parsed `SegmentHeader` and a reference to the original buffer.
/// Section readers are constructed on-demand from validated offsets, borrowing directly
/// from the buffer with no copying.
///
/// Lifetime `'a` ties the view to the buffer lifetime, preventing use-after-free.
///
/// This is the canonical segment view for the current format.
#[derive(Clone, Copy, Debug)]
pub struct SegmentView<'a> {
    buffer: &'a [u8],
    header: SegmentHeader,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "crate-internal section accessors are unused in the lib build until segment queries are wired"
    )
)]
impl<'a> SegmentView<'a> {
    /// Open a segment using the default (Structural) validation mode.
    ///
    /// Equivalent to `open_with_validation(bytes, ValidationMode::Structural)`.
    ///
    /// # Arguments
    /// * `bytes` - complete segment buffer
    ///
    /// # Returns
    /// `Ok(SegmentView)` if validation succeeds, or `SegmentError` if failed.
    pub fn open(bytes: &'a [u8]) -> Result<Self, SegmentError> {
        Self::open_with_validation(bytes, ValidationMode::Structural)
    }

    /// Open a segment with explicit validation mode (`HeaderOnly`, `Structural`, or `Full`).
    ///
    /// # Validation modes:
    ///
    /// - **`HeaderOnly`:** Validate magic bytes and version only. Does NOT traverse sections or
    ///   validate offsets. A buffer with a valid header but truncated/garbage body will open
    ///   successfully. Cheapest open.
    ///
    /// - **Structural (default):** `HeaderOnly` + validate all offsets are in-bounds and sections
    ///   are ordered/non-overlapping via `header.validate_layout()`.
    ///   Safe and fast for most use cases.
    ///
    /// - **Full:** Structural + footer checksum validation via `Footer::verify()`.
    ///   Detects corruption in the covered byte range [0, `footer_offset`).
    ///
    /// # Arguments
    /// * `bytes` - complete segment buffer
    /// * `mode` - validation rigor level
    ///
    /// # Returns
    /// `Ok(SegmentView)` if validation succeeds, or `SegmentError` if failed.
    ///
    /// # Errors
    /// - `SegmentError::Truncated` if buffer is shorter than required
    /// - `SegmentError::BadMagic` if magic bytes do not match
    /// - `SegmentError::UnsupportedVersion` if version is not supported
    /// - `SegmentError::BadOffset` if offsets are out of bounds (Structural/Full only)
    /// - `SegmentError::BadSectionLayout` if sections are misordered (Structural/Full only)
    /// - `SegmentError::BadChecksum` if footer checksum fails (Full only)
    pub fn open_with_validation(
        bytes: &'a [u8],
        mode: ValidationMode,
    ) -> Result<Self, SegmentError> {
        // All modes: read and validate header (magic, version)
        let header = SegmentHeader::read(bytes)?;

        // Structural and Full modes: validate layout (offsets in-bounds and ordered)
        if mode == ValidationMode::Structural || mode == ValidationMode::Full {
            header.validate_layout(bytes.len() as u64)?;
        }

        // Full mode: validate footer checksum
        if mode == ValidationMode::Full {
            footer::Footer::verify(bytes, header.footer_offset)?;
        }

        Ok(Self {
            buffer: bytes,
            header,
        })
    }

    /// Access the field table section as a borrowed reader.
    ///
    /// Constructs a `FieldTableReader` from the validated offset range
    /// [`field_table_offset`, `lexicon_offset`). No copying; the reader borrows directly from the buffer.
    pub(crate) fn field_table(&self) -> Result<FieldTableReader<'a>, SegmentError> {
        FieldTableReader::new(
            self.buffer,
            self.header.field_table_offset,
            self.header.lexicon_offset,
        )
    }

    /// Access the lexicon section as a borrowed reader.
    ///
    /// Constructs a `LexiconReader` from the validated offset range
    /// [`lexicon_offset`, `postings_table_offset`). No copying; the reader borrows directly from the buffer.
    pub(crate) fn lexicon(&self) -> Result<LexiconReader<'a>, SegmentError> {
        LexiconReader::new(
            self.buffer,
            self.header.lexicon_offset,
            self.header.postings_table_offset,
        )
    }

    /// Access the postings table section as a borrowed reader.
    ///
    /// Constructs a `PostingsTableReader` from the validated offset range
    /// [`postings_table_offset`, `postings_data_offset`). No copying; the reader borrows directly from the buffer.
    pub(crate) fn postings_table(&self) -> Result<PostingsTableReader<'a>, SegmentError> {
        PostingsTableReader::new(
            self.buffer,
            self.header.postings_table_offset,
            self.header.postings_data_offset,
        )
    }

    /// Access the postings data section as a borrowed reader.
    ///
    /// Constructs a `PostingsDataReader` from the validated offset range
    /// [`postings_data_offset`, `block_meta_offset`). No copying; the reader borrows directly from the buffer.
    pub(crate) fn postings_data(&self) -> Result<PostingsDataReader<'a>, SegmentError> {
        PostingsDataReader::new(
            self.buffer,
            self.header.postings_data_offset,
            self.header.block_meta_offset,
        )
    }

    /// Access the block metadata section as a borrowed reader (reserved for future releases).
    ///
    /// Constructs a `BlockMetadataReader` from the validated offset range
    /// [`block_meta_offset`, `stored_fields_offset`). In the current format, this section is zero-length.
    /// No copying; the reader borrows directly from the buffer.
    pub(crate) fn block_meta(&self) -> Result<BlockMetadataReader<'a>, SegmentError> {
        BlockMetadataReader::new(
            self.buffer,
            self.header.block_meta_offset,
            self.header.stored_fields_offset,
        )
    }

    /// Returns the total number of documents in this segment.
    ///
    /// This value is stored in the segment header and represents the count of unique document IDs
    /// that were indexed into this segment.
    pub fn document_count(&self) -> u32 {
        self.header.document_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_format::header::{FORMAT_VERSION, MAGIC};

    #[test]
    fn test_reject_truncated_header() {
        let short_buf = [0_u8; 50];
        let result = SegmentView::open(&short_buf);
        assert!(matches!(
            result,
            Err(SegmentError::Truncated {
                needed: 80,
                found: 50
            })
        ));
    }

    #[test]
    fn test_reject_bad_magic() {
        let mut buf = [0_u8; 100];
        buf[0..4].copy_from_slice(&0xDEADBEEF_u32.to_le_bytes());
        buf[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());

        let result = SegmentView::open(&buf);
        assert!(matches!(
            result,
            Err(SegmentError::BadMagic { found: 0xDEADBEEF })
        ));
    }

    #[test]
    fn test_header_only_accepts_truncated_body() {
        // Create header with valid magic and version, rest zeros (truncated)
        let mut buf = [0_u8; 100];

        buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        // Set dummy offsets that are out of bounds
        buf[72..80].copy_from_slice(&1000_u64.to_le_bytes()); // footer_offset = 1000, beyond buffer

        // HeaderOnly should succeed
        let result = SegmentView::open_with_validation(&buf, ValidationMode::HeaderOnly);
        assert!(
            result.is_ok(),
            "HeaderOnly accepts truncated/out-of-bounds offsets"
        );

        // Structural should fail
        let result = SegmentView::open_with_validation(&buf, ValidationMode::Structural);
        assert!(result.is_err(), "Structural rejects out-of-bounds offsets");
    }

    #[test]
    fn test_validation_mode_default_is_structural() {
        assert_eq!(ValidationMode::default(), ValidationMode::Structural);
    }

    #[test]
    fn test_three_modes_are_distinct() {
        assert_ne!(ValidationMode::HeaderOnly, ValidationMode::Structural);
        assert_ne!(ValidationMode::Structural, ValidationMode::Full);
        assert_ne!(ValidationMode::HeaderOnly, ValidationMode::Full);
    }

    #[test]
    fn full_mode_detects_content_corruption_structural_does_not() {
        use crate::memory::{FieldMetadata, PostingEntry, TermEntry};
        use crate::segment_format::writer::write_segment;
        use alloc::collections::{BTreeMap, BTreeSet};
        use alloc::string::String;
        use alloc::vec;
        use leit_core::FieldId;
        use leit_text::FieldAnalyzers;

        // Build a minimal test index
        let documents = BTreeSet::from([0, 1]);
        let field_id = FieldId::new(1);

        let mut field_names = BTreeMap::new();
        field_names.insert(String::from("text"), field_id);

        let mut field_stats = BTreeMap::new();
        field_stats.insert(
            field_id,
            FieldMetadata {
                field_id,
                doc_count: 2,
                total_terms: 2,
            },
        );

        let mut terms_to_ids = BTreeMap::new();
        let mut term_entries = vec![];

        terms_to_ids.insert((field_id, String::from("hello")), leit_core::TermId::new(0));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(0),
            term: String::from("hello"),
        });

        terms_to_ids.insert((field_id, String::from("world")), leit_core::TermId::new(1));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(1),
            term: String::from("world"),
        });

        let mut postings = BTreeMap::new();
        postings.insert(
            leit_core::TermId::new(0),
            vec![PostingEntry {
                doc_id: 0,
                term_freq: 1,
            }],
        );
        postings.insert(
            leit_core::TermId::new(1),
            vec![PostingEntry {
                doc_id: 1,
                term_freq: 1,
            }],
        );

        let mut posting_blocks = BTreeMap::new();
        posting_blocks.insert(leit_core::TermId::new(0), vec![]);
        posting_blocks.insert(leit_core::TermId::new(1), vec![]);

        let field_doc_lengths = BTreeMap::new();

        let index = crate::memory::InMemoryIndex::new(
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

        // Write a valid segment
        let segment = write_segment(&index).expect("write_segment should succeed");

        // Verify it opens successfully in all three modes
        assert!(
            SegmentView::open_with_validation(&segment, ValidationMode::HeaderOnly).is_ok(),
            "HeaderOnly should accept valid segment"
        );
        assert!(
            SegmentView::open_with_validation(&segment, ValidationMode::Structural).is_ok(),
            "Structural should accept valid segment"
        );
        assert!(
            SegmentView::open_with_validation(&segment, ValidationMode::Full).is_ok(),
            "Full should accept valid segment"
        );

        // Corrupt a byte in the postings_data region (not the footer)
        // The footer is at segment.len() - 4, so corrupt somewhere in the middle
        let mut corrupted = segment.clone();
        let corruption_offset = 150; // In the postings_data or postings_table range
        if corruption_offset < corrupted.len() - 4 {
            corrupted[corruption_offset] ^= 0xFF;

            // Structural mode should still accept the corrupted segment (only validates layout, not content)
            let result_structural =
                SegmentView::open_with_validation(&corrupted, ValidationMode::Structural);
            assert!(
                result_structural.is_ok(),
                "Structural mode should accept corrupted content (only validates structure)"
            );

            // Full mode should reject the corrupted segment (validates checksum)
            let result_full = SegmentView::open_with_validation(&corrupted, ValidationMode::Full);
            assert!(
                matches!(result_full, Err(SegmentError::BadChecksum { .. })),
                "Full mode should detect content corruption via checksum"
            );
        }
    }

    #[test]
    fn all_accessors_are_callable_on_valid_segment() {
        use crate::memory::{FieldMetadata, PostingEntry, TermEntry};
        use crate::segment_format::writer::write_segment;
        use alloc::collections::{BTreeMap, BTreeSet};
        use alloc::string::String;
        use alloc::vec;
        use leit_core::FieldId;
        use leit_text::FieldAnalyzers;

        // Build a minimal test index
        let documents = BTreeSet::from([0, 1]);
        let field_id = FieldId::new(1);

        let mut field_names = BTreeMap::new();
        field_names.insert(String::from("text"), field_id);

        let mut field_stats = BTreeMap::new();
        field_stats.insert(
            field_id,
            FieldMetadata {
                field_id,
                doc_count: 2,
                total_terms: 2,
            },
        );

        let mut terms_to_ids = BTreeMap::new();
        let mut term_entries = vec![];

        terms_to_ids.insert((field_id, String::from("hello")), leit_core::TermId::new(0));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(0),
            term: String::from("hello"),
        });

        terms_to_ids.insert((field_id, String::from("world")), leit_core::TermId::new(1));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(1),
            term: String::from("world"),
        });

        let mut postings = BTreeMap::new();
        postings.insert(
            leit_core::TermId::new(0),
            vec![PostingEntry {
                doc_id: 0,
                term_freq: 1,
            }],
        );
        postings.insert(
            leit_core::TermId::new(1),
            vec![PostingEntry {
                doc_id: 1,
                term_freq: 1,
            }],
        );

        let mut posting_blocks = BTreeMap::new();
        posting_blocks.insert(leit_core::TermId::new(0), vec![]);
        posting_blocks.insert(leit_core::TermId::new(1), vec![]);

        let field_doc_lengths = BTreeMap::new();

        let index = crate::memory::InMemoryIndex::new(
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

        // Write a valid segment
        let segment = write_segment(&index).expect("write_segment should succeed");

        // Open with Structural mode (default, zero-copy validation)
        let view = SegmentView::open(&segment).expect("open should succeed");

        // Call all 5 accessors; each should succeed and return a valid reader
        let ft = view
            .field_table()
            .expect("field_table accessor should succeed");
        let lex = view.lexicon().expect("lexicon accessor should succeed");
        let pt = view
            .postings_table()
            .expect("postings_table accessor should succeed");
        let pd = view
            .postings_data()
            .expect("postings_data accessor should succeed");
        let bm = view
            .block_meta()
            .expect("block_meta accessor should succeed");

        // Verify basic sanity: readers are non-null and can be read
        // (We don't deeply validate the data here, just confirm accessors work and return readers)
        let _ = ft;
        let _ = lex;
        let _ = pt;
        let _ = pd;
        let _ = bm;
    }
}
