// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Segment round-trip and validation tests for `leit-index` (Phase 2+ DEC-05 format).

use leit_core::FieldId;
use leit_index::{InMemoryIndexBuilder, SegmentError, SegmentView, ValidationMode};
use leit_text::{Analyzer, FieldAnalyzers, UnicodeNormalizer, WhitespaceTokenizer};

fn test_analyzers() -> FieldAnalyzers {
    let mut analyzers = FieldAnalyzers::new();
    let analyzer =
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new());
    analyzers.set(FieldId::new(1), analyzer);
    analyzers
}

#[test]
fn roundtrip_segment_view_opens_written_segment() {
    let mut builder = InMemoryIndexBuilder::new(test_analyzers());
    builder
        .index_document(1, &[(FieldId::new(1), "Rust Retrieval")])
        .expect("document 1 should index");
    builder
        .index_document(2, &[(FieldId::new(1), "Rust Systems")])
        .expect("document 2 should index");
    let index = builder.build_index();

    let bytes = index
        .to_segment_bytes()
        .expect("segment export should work");

    // Public contract: a written segment opens and passes Structural validation (offsets in-bounds
    // and ordered) and Full validation (footer checksum). Section-content round-trip (field/term/
    // postings byte equality) is covered by the in-crate `segment_format` unit tests via the
    // `pub(crate)` section accessors, which are not part of the public surface yet.
    let view = SegmentView::open(&bytes).expect("written segment should reopen (Structural)");
    assert_eq!(view.document_count(), 2, "segment should have 2 documents");
    SegmentView::open_with_validation(&bytes, ValidationMode::Full)
        .expect("written segment should pass Full validation");
}

#[test]
fn segment_view_rejects_invalid_magic() {
    let mut buf = [0_u8; 100];
    buf[4..8].copy_from_slice(&1_u32.to_le_bytes()); // valid version, bad magic
    let err = SegmentView::open(&buf).expect_err("bad magic should fail");
    assert!(matches!(err, SegmentError::BadMagic { .. }));
}

#[test]
fn segment_view_rejects_truncated_header() {
    let short_buf = [0_u8; 50];
    let err = SegmentView::open(&short_buf).expect_err("short header should fail");
    assert!(matches!(err, SegmentError::Truncated { .. }));
}

#[test]
fn segment_view_accepts_structural_by_default() {
    let mut builder = InMemoryIndexBuilder::new(test_analyzers());
    builder
        .index_document(1, &[(FieldId::new(1), "Rust Retrieval")])
        .expect("document should index");
    let index = builder.build_index();

    let bytes = index
        .to_segment_bytes()
        .expect("segment export should work");

    // Default open should succeed (Structural mode)
    let view = SegmentView::open(&bytes).expect("open should use Structural mode by default");
    assert_eq!(view.document_count(), 1, "segment should have 1 document");
}

#[test]
fn segment_view_accepts_all_validation_modes_on_valid_segment() {
    let mut builder = InMemoryIndexBuilder::new(test_analyzers());
    builder
        .index_document(1, &[(FieldId::new(1), "Rust Retrieval")])
        .expect("document should index");
    let index = builder.build_index();

    let bytes = index
        .to_segment_bytes()
        .expect("segment export should work");

    // All three modes should accept a valid segment
    let _view_header = SegmentView::open_with_validation(&bytes, ValidationMode::HeaderOnly)
        .expect("HeaderOnly should accept valid segment");
    let _view_structural = SegmentView::open_with_validation(&bytes, ValidationMode::Structural)
        .expect("Structural should accept valid segment");
    let _view_full = SegmentView::open_with_validation(&bytes, ValidationMode::Full)
        .expect("Full should accept valid segment");
}

#[test]
fn segment_view_rejects_corrupted_checksum_in_full_mode() {
    let mut builder = InMemoryIndexBuilder::new(test_analyzers());
    builder
        .index_document(1, &[(FieldId::new(1), "Rust Retrieval")])
        .expect("document should index");
    let index = builder.build_index();

    let mut bytes = index
        .to_segment_bytes()
        .expect("segment export should work");

    // Corrupt a byte in the middle of the segment (not the footer)
    let corruption_offset = 150;
    if corruption_offset < bytes.len() - 4 {
        bytes[corruption_offset] ^= 0xFF;

        // Structural mode should still accept (only validates layout, not checksum)
        let result_structural =
            SegmentView::open_with_validation(&bytes, ValidationMode::Structural);
        assert!(
            result_structural.is_ok(),
            "Structural mode should accept corrupted content (only validates structure)"
        );

        // Full mode should reject (validates checksum)
        let result_full = SegmentView::open_with_validation(&bytes, ValidationMode::Full);
        assert!(
            matches!(result_full, Err(SegmentError::BadChecksum { .. })),
            "Full mode should detect content corruption via checksum"
        );
    }
}

#[test]
fn builder_failure_does_not_poison_document_id() {
    let mut builder = InMemoryIndexBuilder::new(test_analyzers());

    let err = builder
        .index_document(7, &[(FieldId::new(2), "missing analyzer")])
        .expect_err("unknown field analyzer should fail");
    assert!(
        matches!(err, leit_index::IndexError::MissingAnalyzer(field) if field == FieldId::new(2))
    );

    builder
        .index_document(7, &[(FieldId::new(1), "retry succeeds")])
        .expect("retrying the same document id after a failed add should work");
}

// NOTE: the deprecated `DirectorySegmentView` shim is exercised by an in-crate test in
// `segment.rs` (which can call the `pub(crate)` legacy `encode_segment` to build a real
// directory-format buffer). An external integration test cannot construct legacy bytes
// because the legacy encoder is crate-internal.
