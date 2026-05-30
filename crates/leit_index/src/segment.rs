// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::ops::Range;

use crate::error::SegmentError;

const MAGIC: [u8; 4] = *b"LSEG";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 24;
const DIRECTORY_ENTRY_LEN: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SectionRef {
    offset: usize,
    len: usize,
}

impl SectionRef {
    const fn range(self) -> Range<usize> {
        self.offset..self.offset.saturating_add(self.len)
    }
}

/// Known section kinds in the legacy directory segment format.
///
/// This enum is deprecated; it remains frozen for compatibility with legacy
/// directory-format segments. New code should use the current `SegmentView` instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
#[deprecated(
    since = "0.2.0",
    note = "use the new fixed-header segment view; the directory format is frozen and will be removed in a future release"
)]
pub enum SectionKind {
    /// Field-aware term dictionary entries.
    TermDictionary = 1,
    /// Field-level document and token statistics.
    FieldMetadata = 2,
    /// Posting list metadata and offsets.
    PostingsMetadata = 3,
    /// Posting list payload bytes.
    PostingsPayload = 4,
}

#[expect(
    deprecated,
    reason = "SectionKind impl is part of deprecated legacy directory-format shim"
)]
impl SectionKind {
    const ALL: [Self; 4] = [
        Self::TermDictionary,
        Self::FieldMetadata,
        Self::PostingsMetadata,
        Self::PostingsPayload,
    ];

    pub(crate) const fn as_u32(self) -> u32 {
        self as u32
    }

    const fn slot(self) -> usize {
        match self {
            Self::TermDictionary => 0,
            Self::FieldMetadata => 1,
            Self::PostingsMetadata => 2,
            Self::PostingsPayload => 3,
        }
    }

    const fn from_u32(value: u32) -> Result<Self, SegmentError> {
        match value {
            1 => Ok(Self::TermDictionary),
            2 => Ok(Self::FieldMetadata),
            3 => Ok(Self::PostingsMetadata),
            4 => Ok(Self::PostingsPayload),
            _ => Err(SegmentError::InvalidSectionKind(value)),
        }
    }
}

/// A validated borrowed view over a serialized segment buffer (legacy directory format).
///
/// This struct is deprecated; it remains frozen for compatibility with legacy
/// directory-format segments. New code should use the current `SegmentView` instead.
///
/// The directory format is no longer actively developed and will be removed in a future release.
#[derive(Clone, Debug)]
#[deprecated(
    since = "0.2.0",
    note = "use the new fixed-header segment view; the directory format is frozen and will be removed in a future release"
)]
pub struct DirectorySegmentView<'a> {
    bytes: &'a [u8],
    document_count: u32,
    term_count: u32,
    field_count: u32,
    sections: [Option<SectionRef>; 4],
}

#[expect(
    deprecated,
    reason = "DirectorySegmentView impl uses its own deprecated struct"
)]
impl<'a> DirectorySegmentView<'a> {
    /// Open and validate a borrowed segment buffer.
    pub fn open(bytes: &'a [u8]) -> Result<Self, SegmentError> {
        if bytes.len() < HEADER_LEN {
            return Err(SegmentError::TruncatedHeader);
        }
        if bytes[0..4] != MAGIC {
            return Err(SegmentError::InvalidMagic);
        }

        let version = read_u16(bytes, 4)?;
        if version != VERSION {
            return Err(SegmentError::UnsupportedVersion {
                found: version as u32,
                expected: VERSION as u32,
            });
        }

        let document_count = read_u32(bytes, 8)?;
        let term_count = read_u32(bytes, 12)?;
        let field_count = read_u32(bytes, 16)?;
        let section_count =
            usize::try_from(read_u32(bytes, 20)?).map_err(|_| SegmentError::TruncatedDirectory)?;

        let directory_len = section_count
            .checked_mul(DIRECTORY_ENTRY_LEN)
            .ok_or(SegmentError::TruncatedDirectory)?;
        let directory_end = HEADER_LEN
            .checked_add(directory_len)
            .ok_or(SegmentError::TruncatedDirectory)?;
        if bytes.len() < directory_end {
            return Err(SegmentError::TruncatedDirectory);
        }

        let mut sections = [None; 4];
        let mut cursor = HEADER_LEN;
        for _ in 0..section_count {
            let kind = SectionKind::from_u32(read_u32(bytes, cursor)?)?;
            let offset_cursor = cursor
                .checked_add(4)
                .ok_or(SegmentError::TruncatedDirectory)?;
            let len_cursor = cursor
                .checked_add(8)
                .ok_or(SegmentError::TruncatedDirectory)?;
            let offset = usize::try_from(read_u32(bytes, offset_cursor)?)
                .map_err(|_| SegmentError::OutOfBoundsSection(kind))?;
            let len = usize::try_from(read_u32(bytes, len_cursor)?)
                .map_err(|_| SegmentError::OutOfBoundsSection(kind))?;
            cursor = cursor
                .checked_add(DIRECTORY_ENTRY_LEN)
                .ok_or(SegmentError::TruncatedDirectory)?;

            let slot = kind.slot();
            if sections[slot].is_some() {
                return Err(SegmentError::DuplicateSection(kind));
            }

            let end = offset
                .checked_add(len)
                .ok_or(SegmentError::OutOfBoundsSection(kind))?;
            if offset < directory_end || end > bytes.len() {
                return Err(SegmentError::OutOfBoundsSection(kind));
            }

            sections[slot] = Some(SectionRef { offset, len });
        }

        for kind in SectionKind::ALL {
            if sections[kind.slot()].is_none() {
                return Err(SegmentError::MissingSection(kind));
            }
        }

        for first_kind in SectionKind::ALL {
            let first = sections[first_kind.slot()].expect("required section validated");
            for second_kind in SectionKind::ALL {
                if first_kind == second_kind {
                    continue;
                }
                let second = sections[second_kind.slot()].expect("required section validated");
                if ranges_overlap(first.range(), second.range()) {
                    return Err(SegmentError::OverlappingSections {
                        first: first_kind,
                        second: second_kind,
                    });
                }
            }
        }

        Ok(Self {
            bytes,
            document_count,
            term_count,
            field_count,
            sections,
        })
    }

    /// Number of indexed documents encoded in the segment.
    pub const fn document_count(&self) -> u32 {
        self.document_count
    }

    /// Number of unique field-aware terms encoded in the segment.
    pub const fn term_count(&self) -> u32 {
        self.term_count
    }

    /// Number of indexed fields encoded in the segment.
    pub const fn field_count(&self) -> u32 {
        self.field_count
    }

    /// Whether the segment contains the requested section.
    pub const fn has_section(&self, kind: SectionKind) -> bool {
        self.sections[kind.slot()].is_some()
    }

    /// Borrow the raw bytes for a section.
    pub fn section_bytes(&self, kind: SectionKind) -> Option<&'a [u8]> {
        let section = self.sections[kind.slot()]?;
        self.bytes.get(section.range())
    }
}

pub(crate) const fn magic() -> [u8; 4] {
    MAGIC
}

pub(crate) const fn version() -> u16 {
    VERSION
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, SegmentError> {
    let end = offset.checked_add(2).ok_or(SegmentError::TruncatedHeader)?;
    let raw: [u8; 2] = bytes
        .get(offset..end)
        .ok_or(SegmentError::TruncatedHeader)?
        .try_into()
        .map_err(|_| SegmentError::TruncatedHeader)?;
    Ok(u16::from_le_bytes(raw))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, SegmentError> {
    let end = offset.checked_add(4).ok_or(SegmentError::TruncatedHeader)?;
    let raw: [u8; 4] = bytes
        .get(offset..end)
        .ok_or(SegmentError::TruncatedHeader)?
        .try_into()
        .map_err(|_| SegmentError::TruncatedHeader)?;
    Ok(u32::from_le_bytes(raw))
}

const fn ranges_overlap(a: Range<usize>, b: Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

#[cfg(test)]
mod deprecated_shim_tests {
    //! Keeps the frozen legacy directory-format reader (`DirectorySegmentView`) exercised:
    //! it must still read a legacy directory buffer produced by the legacy `encode_segment`.
    #![expect(
        deprecated,
        reason = "intentionally exercises the deprecated legacy directory-format shim"
    )]

    use super::{DirectorySegmentView, SectionKind};
    use crate::InMemoryIndexBuilder;
    use crate::codec::encode_segment;
    use leit_core::FieldId;
    use leit_text::{Analyzer, FieldAnalyzers, UnicodeNormalizer, WhitespaceTokenizer};

    fn analyzers() -> FieldAnalyzers {
        let mut a = FieldAnalyzers::new();
        a.set(
            FieldId::new(1),
            Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new()),
        );
        a
    }

    #[test]
    fn directory_segment_view_reads_legacy_buffer() {
        let mut builder = InMemoryIndexBuilder::new(analyzers());
        builder
            .index_document(1, &[(FieldId::new(1), "legacy rust")])
            .expect("doc 1 indexes");
        builder
            .index_document(2, &[(FieldId::new(1), "legacy systems")])
            .expect("doc 2 indexes");
        let index = builder.build_index();

        // Legacy directory-format bytes (the frozen legacy encoder).
        let legacy = encode_segment(&index).expect("legacy encode should succeed");

        // The deprecated directory reader still parses them.
        let view =
            DirectorySegmentView::open(&legacy).expect("directory view should open legacy bytes");
        assert_eq!(view.document_count(), 2);
        assert_eq!(view.field_count(), 1);
        assert!(view.term_count() >= 2);
        assert!(view.has_section(SectionKind::TermDictionary));
        assert!(view.has_section(SectionKind::PostingsPayload));
    }
}
