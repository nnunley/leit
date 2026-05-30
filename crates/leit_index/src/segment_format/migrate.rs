// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Segment migration tooling: version dispatch framework and format upgrade path.
//!
//! This module provides a single entry point for migrating segments between versions.
//! It validates the version, dispatches to the appropriate migration handler, and returns
//! the migrated bytes (identity migration for v1; future versions add handlers here).
//!
//! **Supported versions:** v1 only (current format).
//!
//! **Migration policy:** The v1 segment format is supported unbounded with no planned sunset.
//! The migration framework establishes the dispatch mechanism for future versions: when a v2 is
//! introduced, this module supports reading a sliding window of versions (e.g. the current version
//! and its predecessor) and provides a version-specific rewrite handler for each older version.

use alloc::borrow::Cow;
use core::fmt;

use crate::error::SegmentError;
use crate::segment_format::header::{FORMAT_VERSION, SegmentHeader};

/// Set of segment format versions that are currently supported for reading.
///
/// This constant lists the versions a reader can accept. Unknown versions outside
/// this set are cleanly rejected. When a new format version is introduced, this set
/// is updated alongside a new migration handler in `migrate_to_current()`.
const SUPPORTED_VERSIONS: &[u32] = &[1];

/// Migrate a segment buffer to the current format version.
///
/// This is the main entry point for segment migration. It reads the segment header,
/// validates the version, and dispatches to the appropriate migration handler:
/// - If the segment is already at the current version (v1), it structurally validates
///   the segment and returns the bytes unchanged.
/// - If the segment is at an older but supported version, it applies the version-specific
///   migration handler.
/// - If the segment version is unknown or unsupported, it returns `SegmentError::UnsupportedVersion`.
///
/// # Arguments
/// * `bytes` - complete segment buffer
///
/// # Returns
/// - `Ok(Cow::Borrowed(&[u8]))` if the segment is already at the current version
///   (identity migration), with structural validation performed.
/// - `Ok(Cow::Owned(Vec<u8>))` if the segment required migration (future versions).
/// - `Err(SegmentError)` if validation or migration fails.
///
/// # Errors
/// - `SegmentError::Truncated` if the buffer is too short for the header.
/// - `SegmentError::BadMagic` if the magic bytes do not match.
/// - `SegmentError::UnsupportedVersion` if the version is not supported (includes
///   the found version and expected/supported versions in the error message).
/// - `SegmentError::BadOffset`, `SegmentError::BadSectionLayout` if structural validation fails.
pub fn migrate_to_current(bytes: &[u8]) -> Result<Cow<'_, [u8]>, SegmentError> {
    // Read and validate header (magic, version).
    let header = SegmentHeader::read(bytes)?;

    // Dispatch on version.
    match header.version {
        1 => {
            // v1 is current. Structurally validate and return unchanged.
            header.validate_layout(bytes.len() as u64)?;
            Ok(Cow::Borrowed(bytes))
        }
        unsupported => {
            // Unknown or unsupported version.
            Err(SegmentError::UnsupportedVersion {
                found: unsupported,
                expected: FORMAT_VERSION,
            })
        }
    }
}

/// Check if a version is in the supported set.
///
/// Used internally for informative error messages and future validation logic.
#[inline]
#[expect(
    dead_code,
    reason = "reserved for future version-range error reporting"
)]
fn is_supported_version(version: u32) -> bool {
    SUPPORTED_VERSIONS.contains(&version)
}

/// Describe the supported versions in a human-readable format.
///
/// Useful for error reporting and documentation.
#[inline]
#[expect(dead_code, reason = "reserved for future error messages")]
fn supported_versions_description() -> VersionRangeDescription {
    VersionRangeDescription {
        current: FORMAT_VERSION,
        supported_count: SUPPORTED_VERSIONS.len(),
    }
}

/// A compact representation of the version support window for error messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VersionRangeDescription {
    /// The current format version.
    current: u32,
    /// The number of supported versions (e.g., "1" for v1-only).
    supported_count: usize,
}

impl fmt::Display for VersionRangeDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "version {} (current); {} version(s) supported",
            self.current, self.supported_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::{BTreeMap, BTreeSet};
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use leit_core::FieldId;
    use leit_text::FieldAnalyzers;

    use crate::memory::{FieldMetadata, InMemoryIndex, PostingEntry, TermEntry};
    use crate::segment_format::writer::write_segment;

    /// Helper: build a minimal v1 segment with one field and a few terms.
    fn build_minimal_v1_segment() -> Vec<u8> {
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
                total_terms: 3,
            },
        );

        let mut terms_to_ids = BTreeMap::new();
        let mut term_entries = vec![];

        // Term "hello" (term_id = 0)
        terms_to_ids.insert((field_id, String::from("hello")), leit_core::TermId::new(0));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(0),
            term: String::from("hello"),
        });

        // Term "world" (term_id = 1)
        terms_to_ids.insert((field_id, String::from("world")), leit_core::TermId::new(1));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(1),
            term: String::from("world"),
        });

        // Build postings
        let mut postings: BTreeMap<leit_core::TermId, Vec<PostingEntry>> = BTreeMap::new();
        postings.insert(
            leit_core::TermId::new(0),
            vec![
                PostingEntry {
                    doc_id: 0,
                    term_freq: 1,
                },
                PostingEntry {
                    doc_id: 1,
                    term_freq: 1,
                },
            ],
        );
        postings.insert(
            leit_core::TermId::new(1),
            vec![PostingEntry {
                doc_id: 0,
                term_freq: 1,
            }],
        );

        let field_doc_lengths = BTreeMap::new();
        let posting_blocks = BTreeMap::new();

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

        write_segment(&index).expect("failed to write segment")
    }

    #[test]
    fn migrate_v1_is_identity_roundtrip() {
        // Build a v1 segment.
        let original_bytes = build_minimal_v1_segment();

        // Migrate to current (should be identity).
        let migrated =
            migrate_to_current(&original_bytes).expect("migration should succeed for v1 segment");

        // Assert the result is Borrowed (identity, no allocation).
        match migrated {
            Cow::Borrowed(_) => {
                // Good: identity migration.
            }
            Cow::Owned(_) => {
                panic!("v1 migration should be identity (Borrowed), not owned");
            }
        }

        // Assert the bytes are identical.
        assert_eq!(migrated.as_ref(), original_bytes.as_slice());

        // Verify the migrated bytes open and validate as a v1 segment.
        let view = crate::segment_format::view::SegmentView::open(&migrated)
            .expect("migrated segment should open");

        // Check counts match the original.
        assert_eq!(view.document_count(), 2);
    }

    #[test]
    fn migrate_rejects_unknown_version() {
        // Build a v1 segment.
        let mut original_bytes = build_minimal_v1_segment();

        // Mutate the version field in the header (bytes 4-7) to an unsupported value.
        let unsupported_version = 99_u32;
        original_bytes[4..8].copy_from_slice(&unsupported_version.to_le_bytes());

        // Attempt migration; should fail with UnsupportedVersion.
        let result = migrate_to_current(&original_bytes);
        match result {
            Err(SegmentError::UnsupportedVersion { found, expected }) => {
                assert_eq!(found, 99);
                assert_eq!(expected, FORMAT_VERSION);
            }
            _ => {
                panic!(
                    "expected UnsupportedVersion error with found=99, expected={}; got {:?}",
                    FORMAT_VERSION, result
                );
            }
        }

        // Also verify that SegmentView::open rejects the same mutated buffer.
        let view_result = crate::segment_format::view::SegmentView::open(&original_bytes);
        match view_result {
            Err(SegmentError::UnsupportedVersion { found, expected }) => {
                assert_eq!(found, 99);
                assert_eq!(expected, FORMAT_VERSION);
            }
            _ => {
                panic!(
                    "expected SegmentView::open to also reject with UnsupportedVersion; got {:?}",
                    view_result
                );
            }
        }
    }

    #[test]
    fn migrate_rejects_bad_magic() {
        // Build a v1 segment.
        let mut original_bytes = build_minimal_v1_segment();

        // Mutate the magic field to something invalid.
        let bad_magic = 0xDEADBEEF_u32;
        original_bytes[0..4].copy_from_slice(&bad_magic.to_le_bytes());

        // Attempt migration; should fail with BadMagic.
        let result = migrate_to_current(&original_bytes);
        match result {
            Err(SegmentError::BadMagic { found }) => {
                assert_eq!(found, bad_magic);
            }
            _ => {
                panic!(
                    "expected BadMagic error with found=0xDEADBEEF; got {:?}",
                    result
                );
            }
        }
    }

    #[test]
    fn migrate_rejects_truncated_buffer() {
        // Create a buffer shorter than the header size.
        let short_buf = [0_u8; 40];

        let result = migrate_to_current(&short_buf);
        match result {
            Err(SegmentError::Truncated { needed, found }) => {
                assert_eq!(needed, 80);
                assert_eq!(found, 40);
            }
            _ => {
                panic!("expected Truncated error; got {:?}", result);
            }
        }
    }

    #[test]
    fn migrate_enforces_structural_validation() {
        // Build a v1 segment and truncate it mid-section (after header but before end).
        let original_bytes = build_minimal_v1_segment();

        // Truncate to just past the header: create a buffer with a valid header
        // but truncated sections.
        if original_bytes.len() > 80 + 100 {
            let truncated = &original_bytes[..80 + 100];
            let result = migrate_to_current(truncated);

            // Should fail because the sections are out of bounds.
            match result {
                Err(SegmentError::BadOffset { .. }) => {
                    // Good: structural validation caught it.
                }
                Err(SegmentError::BadSectionLayout) => {
                    // Also acceptable: layout validation.
                }
                _ => {
                    panic!(
                        "expected BadOffset or BadSectionLayout for truncated segment; got {:?}",
                        result
                    );
                }
            }
        }
    }
}
