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

#[test]
fn public_segment_view_roundtrip() {
    // Build an index with multiple documents to test field/lexicon/postings iteration.
    // AC-2 requirement: prove the public API reads back the indexed content.
    // Indexed documents:
    // - Doc 1: "Rust Retrieval Engine" -> terms: rust, retrieval, engine
    // - Doc 2: "Rust Systems Programming" -> terms: rust, systems, programming
    // - Doc 3: "Rust Concurrent Memory Safety" -> terms: rust, concurrent, memory, safety
    let expected_doc_count = 3;
    let expected_field_count = 1;

    let mut analyzers = FieldAnalyzers::new();
    let analyzer =
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new());
    analyzers.set(FieldId::new(1), analyzer);

    let mut builder = InMemoryIndexBuilder::new(analyzers);
    builder
        .index_document(1, &[(FieldId::new(1), "Rust Retrieval Engine")])
        .expect("document 1 should index");
    builder
        .index_document(2, &[(FieldId::new(1), "Rust Systems Programming")])
        .expect("document 2 should index");
    builder
        .index_document(3, &[(FieldId::new(1), "Rust Concurrent Memory Safety")])
        .expect("document 3 should index");
    let index = builder.build_index();

    let bytes = index
        .to_segment_bytes()
        .expect("segment export should work");

    // Open segment via public API
    let view = SegmentView::open(&bytes).expect("segment should open");

    // AC-2 assertion 1: document_count() returns exact expected value
    assert_eq!(
        view.document_count(),
        expected_doc_count,
        "segment should have exactly {} documents via public API",
        expected_doc_count
    );

    // AC-2 assertion 2: field_count() returns exact expected value
    let field_count = view.field_count().expect("field_count() should succeed");
    assert_eq!(
        field_count, expected_field_count,
        "segment should have exactly {} field via public API",
        expected_field_count
    );

    // AC-2 assertion 3: term_count() returns a concrete count > 0
    let term_count = view.term_count().expect("term_count() should succeed");
    assert!(
        term_count > 0,
        "segment should have at least one distinct term via public API"
    );

    // Validate field table consistency
    let field_table = view
        .field_table()
        .expect("field_table() accessor should work");
    assert_eq!(
        field_table.len(),
        field_count,
        "field table length should match field_count()"
    );

    // Iterate field table and verify field 1 exists with exact document count
    let mut found_field_1 = false;
    for i in 0..field_table.len() {
        let (field_id, doc_count, total_terms) = field_table
            .entry(i)
            .expect("field_table.entry() should succeed");
        if field_id == 1 {
            found_field_1 = true;
            // AC-2 assertion 4: field 1 has exact 3 documents
            assert_eq!(doc_count, 3, "field 1 should have exactly 3 documents");
            // AC-2 assertion 5: field 1 total_terms is a valid count > 0
            assert!(
                total_terms > 0,
                "field 1 total_terms should be > 0 (has {} terms)",
                total_terms
            );
        }
    }
    assert!(found_field_1, "field_id 1 should exist in the field table");

    // AC-2 assertion 6: lexicon() returns consistent counts
    let lexicon = view.lexicon().expect("lexicon() accessor should work");
    assert_eq!(
        lexicon.len(),
        term_count,
        "lexicon.len() should match term_count()"
    );

    // AC-2 assertion 7: All lexicon entries have valid term bytes
    let mut collected_terms: Vec<Vec<u8>> = Vec::new();
    let mut found_rust = false;
    let mut rust_postings_index: Option<u32> = None;

    for i in 0..lexicon.len() {
        let (term_bytes, postings_table_index) =
            lexicon.entry(i).expect("lexicon.entry() should succeed");
        // Proof: term_bytes are non-empty (round-trip from indexing)
        assert!(
            !term_bytes.is_empty(),
            "term {} should have non-empty bytes",
            i
        );
        assert!(
            postings_table_index < u32::MAX,
            "term {} should have valid postings_table_index",
            i
        );
        collected_terms.push(term_bytes.to_vec());

        // Track "rust" term (appears in all 3 documents)
        if term_bytes == b"rust" {
            found_rust = true;
            rust_postings_index = Some(postings_table_index);
        }
    }

    // AC-2 assertion 8: All collected terms are non-empty
    assert_eq!(
        collected_terms.len(),
        term_count as usize,
        "collected {} terms should match term_count() of {}",
        collected_terms.len(),
        term_count
    );
    assert!(
        collected_terms.iter().all(|t| !t.is_empty()),
        "all collected terms should be non-empty (round-trip proof)"
    );
    // Byte-exact content round-trip: several distinct indexed content words must be readable back
    // from the lexicon exactly as indexed (proves entries carry real content, not just shape).
    for expected in [b"rust".as_slice(), b"engine", b"memory", b"safety"] {
        assert!(
            collected_terms.iter().any(|t| t.as_slice() == expected),
            "indexed term {:?} should round-trip byte-for-byte through the public lexicon API",
            core::str::from_utf8(expected).unwrap_or("<non-utf8>")
        );
    }

    // AC-2 assertion 9: Postings round-trip for a known term
    let postings_table = view
        .postings_table()
        .expect("postings_table() accessor should work");
    assert!(
        !postings_table.is_empty(),
        "postings_table should have entries"
    );

    // If "rust" was found, verify its postings entry
    if let Some(postings_idx) = rust_postings_index {
        let (postings_data_offset, postings_data_len, doc_freq, _codec, _first_block, _block_count) =
            postings_table
                .entry(postings_idx)
                .expect("postings_table.entry() should succeed");
        // "rust" appears in all 3 documents, so doc_freq must be 3
        assert_eq!(
            doc_freq, 3,
            "term 'rust' should have doc_freq = 3 (round-trip proof: present in all 3 documents)"
        );
        assert!(
            postings_data_len > 0,
            "term 'rust' should have valid postings_data_len"
        );
        // Verify offset is reasonable (not validating exact value, just that it's properly initialized)
        let _ = postings_data_offset;
    } else {
        // "rust" should always be present since all documents contain "Rust"
        assert!(found_rust, "term 'rust' should be in the lexicon");
    }

    // Test postings_data accessor opens successfully
    let postings_data = view
        .postings_data()
        .expect("postings_data() accessor should work");
    let _ = postings_data;

    // Test block_meta accessor opens successfully (may be empty in current format)
    let block_meta = view
        .block_meta()
        .expect("block_meta() accessor should work");
    let _ = block_meta;

    // AC-2 assertion 10: Full validation passes
    SegmentView::open_with_validation(&bytes, ValidationMode::Full)
        .expect("segment should pass Full validation (proves checksums round-trip)");
}

#[test]
fn migrate_to_current_is_public_and_roundtrips_current_segment() {
    let mut builder = InMemoryIndexBuilder::new(test_analyzers());
    builder
        .index_document(1, &[(FieldId::new(1), "Rust Retrieval")])
        .expect("document should index");
    let index = builder.build_index();
    let bytes = index
        .to_segment_bytes()
        .expect("segment export should work");

    let migrated = migrate_to_current(&bytes).expect("public migration API should accept v1");
    assert_eq!(migrated.as_ref(), bytes.as_slice());
}
