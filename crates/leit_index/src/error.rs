// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Index and segment error types.
//!
//! `SegmentError` retains a set of legacy directory-format variants that reference the deprecated
//! `SectionKind`; the file-level expectation below covers those intentional references (the legacy
//! directory reader is a frozen shim) without resorting to `#[allow]`.
#![expect(
    deprecated,
    reason = "legacy directory-format SegmentError variants intentionally reference the deprecated SectionKind"
)]

use core::fmt;

use leit_query::QueryError;

use crate::segment::SectionKind;

/// Errors produced while building or exporting an in-memory index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexError {
    /// A document ID was indexed more than once.
    DuplicateDocument(u32),
    /// A field was indexed without a registered analyzer.
    MissingAnalyzer(leit_core::FieldId),
    /// Execution required scores but no scorer was supplied.
    MissingScorer,
    /// A size or offset does not fit in the on-disk format.
    ValueOutOfRange,
    /// Structured filter predicates require columnar storage (a future extension).
    UnsupportedFilterPredicate,
    /// Query planning failed.
    Query(QueryError),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateDocument(id) => write!(f, "duplicate document ID: {id}"),
            Self::MissingAnalyzer(field) => {
                write!(f, "missing analyzer for field {}", field.as_u32())
            }
            Self::MissingScorer => write!(f, "execution requires a scorer but none was provided"),
            Self::UnsupportedFilterPredicate => write!(
                f,
                "structured filter predicates require columnar storage (not yet implemented)"
            ),
            Self::ValueOutOfRange => write!(f, "value out of range for on-disk format"),
            Self::Query(err) => write!(f, "query error: {err}"),
        }
    }
}

impl core::error::Error for IndexError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Query(err) => Some(err),
            Self::DuplicateDocument(_)
            | Self::MissingAnalyzer(_)
            | Self::MissingScorer
            | Self::UnsupportedFilterPredicate
            | Self::ValueOutOfRange => None,
        }
    }
}

/// Segment validation mode: determines what checks are performed during segment open.
///
/// - `HeaderOnly`: validate magic, version, and header self-consistency.
/// - `Structural`: (default) additionally validate all offsets are in-bounds and sections
///   are ordered/non-overlapping.
/// - `Full`: additionally validate footer checksum and per-section structural invariants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ValidationMode {
    /// Validate magic and version only. Cheapest open.
    HeaderOnly,
    /// Validate header + offset in-bounds + section ordering (default). Safe and fast.
    #[default]
    Structural,
    /// Validate everything including footer checksum and per-section invariants.
    Full,
}

/// Errors produced while opening or validating a borrowed segment.
///
/// All variants carry structured context (no heap-allocated strings) for zero-copy error handling.
/// Suitable for `no_std` environments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentError {
    /// Buffer truncated: needed at least `needed` bytes but found only `found`.
    Truncated {
        /// Number of bytes required.
        needed: usize,
        /// Number of bytes actually found.
        found: usize,
    },
    /// Magic bytes mismatch: found `found` instead of expected magic.
    BadMagic {
        /// Actual magic bytes found in buffer.
        found: u32,
    },
    /// Unsupported version: found `found`, expected `expected`.
    UnsupportedVersion {
        /// Actual version found in buffer.
        found: u32,
        /// Expected version number.
        expected: u32,
    },
    /// Offset out of bounds or non-monotonic: offset `offset` exceeds limit `limit`.
    BadOffset {
        /// Byte offset that fell outside the segment buffer.
        offset: u64,
        /// Segment buffer size limit.
        limit: u64,
    },
    /// Section layout invalid: sections overlap or are mis-ordered (legacy directory format).
    BadSectionLayout,
    /// Block metadata section malformed (reserved for future releases).
    InvalidBlockMeta,
    /// Footer checksum mismatch: expected `expected`, found `found`.
    BadChecksum {
        /// Expected checksum value.
        expected: u32,
        /// Actual checksum computed from buffer.
        found: u32,
    },

    // Legacy directory-format variants (replaced but kept for backward compat)
    /// The buffer does not start with the expected magic bytes. (LEGACY)
    InvalidMagic,
    /// The fixed-size header was truncated. (LEGACY)
    TruncatedHeader,
    /// The section directory was truncated. (LEGACY)
    TruncatedDirectory,
    /// A section kind in the directory is not known to this reader. (LEGACY)
    InvalidSectionKind(u32),
    /// A section appears more than once. (LEGACY)
    DuplicateSection(SectionKind),
    /// A required section is missing. (LEGACY)
    MissingSection(SectionKind),
    /// A section offset/length points outside the buffer. (LEGACY)
    OutOfBoundsSection(SectionKind),
    /// Two declared sections overlap. (LEGACY)
    OverlappingSections {
        /// The first overlapping section.
        first: SectionKind,
        /// The second overlapping section.
        second: SectionKind,
    },
}

impl fmt::Display for SegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { needed, found } => {
                write!(f, "truncated buffer: needed {needed} bytes, found {found}")
            }
            Self::BadMagic { found } => {
                write!(f, "invalid magic bytes: found 0x{found:08x}")
            }
            Self::UnsupportedVersion { found, expected } => {
                write!(f, "unsupported version: found {found}, expected {expected}")
            }
            Self::BadOffset { offset, limit } => {
                write!(
                    f,
                    "offset out of bounds: offset {offset} exceeds limit {limit}"
                )
            }
            Self::BadSectionLayout => {
                write!(f, "section layout invalid: sections overlap or mis-ordered")
            }
            Self::InvalidBlockMeta => {
                write!(f, "block metadata section malformed")
            }
            Self::BadChecksum { expected, found } => {
                write!(
                    f,
                    "checksum mismatch: expected 0x{expected:08x}, found 0x{found:08x}"
                )
            }
            // Legacy directory-format variants
            Self::InvalidMagic => write!(f, "invalid segment magic bytes"),
            Self::TruncatedHeader => write!(f, "truncated segment header"),
            Self::TruncatedDirectory => write!(f, "truncated section directory"),
            Self::InvalidSectionKind(k) => write!(f, "invalid section kind: {k}"),
            Self::DuplicateSection(k) => write!(f, "duplicate section: {k:?}"),
            Self::MissingSection(k) => write!(f, "missing required section: {k:?}"),
            Self::OutOfBoundsSection(k) => write!(f, "out-of-bounds section: {k:?}"),
            Self::OverlappingSections { first, second } => {
                write!(f, "overlapping sections: {first:?} and {second:?}")
            }
        }
    }
}

impl core::error::Error for SegmentError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_mode_variants_exist() {
        let _ = ValidationMode::HeaderOnly;
        let _ = ValidationMode::Structural;
        let _ = ValidationMode::Full;
    }

    #[test]
    fn test_validation_mode_default_is_structural() {
        assert_eq!(ValidationMode::default(), ValidationMode::Structural);
    }

    #[test]
    fn test_validation_mode_can_be_copied() {
        let mode = ValidationMode::Full;
        let mode_copy = mode;
        assert_eq!(mode, mode_copy);
    }

    #[test]
    fn test_segment_error_truncated_structured() {
        let err = SegmentError::Truncated {
            needed: 100,
            found: 50,
        };
        if let SegmentError::Truncated { needed, found } = err {
            assert_eq!(needed, 100);
            assert_eq!(found, 50);
        } else {
            panic!("expected Truncated variant");
        }
    }

    #[test]
    fn test_segment_error_bad_magic_structured() {
        let err = SegmentError::BadMagic { found: 0xDEADBEEF };
        if let SegmentError::BadMagic { found } = err {
            assert_eq!(found, 0xDEADBEEF);
        } else {
            panic!("expected BadMagic variant");
        }
    }

    #[test]
    fn test_segment_error_unsupported_version_structured() {
        let err = SegmentError::UnsupportedVersion {
            found: 99,
            expected: 1,
        };
        if let SegmentError::UnsupportedVersion { found, expected } = err {
            assert_eq!(found, 99);
            assert_eq!(expected, 1);
        } else {
            panic!("expected UnsupportedVersion variant");
        }
    }

    #[test]
    fn test_segment_error_bad_offset_structured() {
        let err = SegmentError::BadOffset {
            offset: 1000,
            limit: 500,
        };
        if let SegmentError::BadOffset { offset, limit } = err {
            assert_eq!(offset, 1000);
            assert_eq!(limit, 500);
        } else {
            panic!("expected BadOffset variant");
        }
    }

    #[test]
    fn test_segment_error_bad_section_layout() {
        let err = SegmentError::BadSectionLayout;
        if let SegmentError::BadSectionLayout = err {
            // success
        } else {
            panic!("expected BadSectionLayout variant");
        }
    }

    #[test]
    fn test_segment_error_invalid_block_meta() {
        let err = SegmentError::InvalidBlockMeta;
        if let SegmentError::InvalidBlockMeta = err {
            // success
        } else {
            panic!("expected InvalidBlockMeta variant");
        }
    }

    #[test]
    fn test_segment_error_truncated_cloneable() {
        let err = SegmentError::Truncated {
            needed: 10,
            found: 5,
        };
        let _cloned = err.clone();
    }

    #[test]
    fn test_segment_error_bad_magic_cloneable() {
        let err = SegmentError::BadMagic { found: 0x12345678 };
        let _cloned = err.clone();
    }

    #[test]
    fn test_segment_error_unsupported_version_cloneable() {
        let err = SegmentError::UnsupportedVersion {
            found: 2,
            expected: 1,
        };
        let _cloned = err.clone();
    }

    #[test]
    fn test_segment_error_bad_offset_cloneable() {
        let err = SegmentError::BadOffset {
            offset: 100,
            limit: 50,
        };
        let _cloned = err.clone();
    }

    #[test]
    fn test_segment_error_bad_section_layout_cloneable() {
        let err = SegmentError::BadSectionLayout;
        let _cloned = err.clone();
    }

    #[test]
    fn test_segment_error_invalid_block_meta_cloneable() {
        let err = SegmentError::InvalidBlockMeta;
        let _cloned = err.clone();
    }
}
