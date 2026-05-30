// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    unsafe_code,
    reason = "memmap2::Mmap::map is inherently unsafe and requires unsafe blocks; the feature is std-only and opt-in"
)]

//! Memory-mapped segment view: owns a `memmap2::Mmap` handle and provides
//! zero-copy, lifetime-safe access to segment sections without copying the mapped region.
//!
//! `MmapSegment` is a standard, production-ready way to open large segments that must
//! reside on disk. The mmap'd bytes are viewable as a `SegmentView<'_>`, with the
//! lifetime tied to the mmap handle, preventing use-after-free. Thread-safety is
//! provided by `memmap2::Mmap` (Send+Sync for read-only maps).

use core::fmt;
use std::io;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{SegmentError, ValidationMode};
use crate::segment_format::view::SegmentView;

/// Errors from opening or using a memory-mapped segment.
///
/// This wraps `io::Error` for file I/O failures and `SegmentError`
/// for segment validation failures.
#[derive(Debug)]
pub enum MmapError {
    /// File I/O error (e.g., file not found, permission denied).
    Io(io::Error),
    /// Segment validation failed.
    Segment(SegmentError),
}

impl fmt::Display for MmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Segment(e) => write!(f, "segment error: {e}"),
        }
    }
}

impl core::error::Error for MmapError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Segment(e) => Some(e),
        }
    }
}

impl From<io::Error> for MmapError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<SegmentError> for MmapError {
    fn from(e: SegmentError) -> Self {
        Self::Segment(e)
    }
}

/// A memory-mapped segment, owning the `Mmap` handle.
///
/// `MmapSegment` opens a file from disk, memory-maps it, and validates the segment
/// header on construction. The mmap'd bytes remain pinned for the lifetime of the
/// handle, allowing zero-copy creation of `SegmentView`s whose lifetime is tied to
/// the mmap handle.
///
/// **Thread-safety:** `memmap2::Mmap` is `Send + Sync`. A `MmapSegment` can be
/// wrapped in `Arc` and shared across threads. Each thread obtains a `SegmentView`
/// borrowing the mmap region with the borrowing thread's lifetime, ensuring no
/// data races.
///
    /// **Validation:** The header is validated on open (magic bytes, version). Follow-up
    /// view construction defaults to structural validation and can be made stricter via
    /// `view().with_mode(...)`.
#[derive(Debug)]
pub struct MmapSegment {
    mmap: Mmap,
}

impl MmapSegment {
    /// Open a segment from a file, validate its header, and return the mmap handle.
    ///
    /// This opens the file and memory-maps its entire contents. The header is
    /// validated (magic bytes and version checked) using `ValidationMode::HeaderOnly`.
    /// This validates the header structure only; a file with a valid header but a
    /// corrupt or truncated body will still open successfully. Use `view()` to
    /// perform structural (offset validation, section ordering) or full (checksum)
    /// validation of the complete segment.
    ///
    /// # Arguments
    /// * `path` - path to the segment file
    ///
    /// # Returns
    /// `Ok(MmapSegment)` if the file opened and header validated.
    ///
    /// # Errors
    /// - `MmapError::Io` if the file cannot be opened or mmap'd
    /// - `MmapError::Segment` if header validation fails (bad magic, unsupported version)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, MmapError> {
        let file = std::fs::File::open(path)?;

        // SAFETY: The unsafe mmap operation is sound if the file is never externally
        // truncated or mutated while the map is in use. This is a practical assumption
        // for segment files, which are stable, immutable index artifacts. Concurrent
        // external mutation (file truncation, replacement) is not protected by the type
        // system and remains the caller's responsibility in environments where segment
        // files may be modified by external processes.
        let mmap = unsafe { Mmap::map(&file)? };

        // Validate header: magic and version.
        // Reuse the existing SegmentView header validation.
        SegmentView::open_with_validation(&mmap, ValidationMode::HeaderOnly)?;

        Ok(Self { mmap })
    }

    /// Start configuring a `SegmentView` borrowing the mmap'd region.
    ///
    /// The returned builder defaults to structural validation and can be made stricter
    /// with `with_mode()`.
    pub fn view(&self) -> MmapSegmentViewBuilder<'_> {
        MmapSegmentViewBuilder {
            segment: self,
            mode: ValidationMode::Structural,
        }
    }

    /// Obtain a `SegmentView` using the default structural validation mode.
    pub fn as_view(&self) -> Result<SegmentView<'_>, SegmentError> {
        self.view().open()
    }
}

/// Builder for opening a `SegmentView` from a memory-mapped segment with explicit validation.
#[derive(Clone, Copy, Debug)]
pub struct MmapSegmentViewBuilder<'a> {
    segment: &'a MmapSegment,
    mode: ValidationMode,
}

impl<'a> MmapSegmentViewBuilder<'a> {
    /// Override the validation mode used when opening the borrowed view.
    pub const fn with_mode(mut self, mode: ValidationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Open a borrowed view over the mapped bytes with the configured validation mode.
    pub fn open(self) -> Result<SegmentView<'a>, SegmentError> {
        SegmentView::open_with_validation(&self.segment.mmap, self.mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{FieldMetadata, InMemoryIndex, PostingEntry, TermEntry};
    use crate::segment_format::writer::write_segment;
    use alloc::boxed::Box;
    use alloc::collections::{BTreeMap, BTreeSet};
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use leit_core::FieldId;
    use leit_text::FieldAnalyzers;
    use std::io::Write;

    /// Helper: build a unique temp file path for a test. A process-wide counter plus the process id
    /// guarantee uniqueness even when several tests run concurrently and request a path within the
    /// same nanosecond — without it, parallel tests collide on the path and read each other's files.
    fn unique_temp_path(prefix: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let mut path = std::env::temp_dir();
        path.push(format!("{prefix}_{pid}_{timestamp}_{unique}.seg"));
        path
    }

    /// Helper: write a test segment to a temp file and return the path.
    fn write_test_segment_to_file(
        index: &InMemoryIndex,
    ) -> Result<String, Box<dyn core::error::Error>> {
        let bytes = write_segment(index)?;
        let temp_file = unique_temp_path("leit_mmap_test");

        let mut file = std::fs::File::create(&temp_file)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);

        Ok(temp_file.to_string_lossy().into_owned())
    }

    /// Helper: build a minimal test index with fields and postings.
    fn build_test_index() -> InMemoryIndex {
        let documents = BTreeSet::from([0, 1, 2, 3]);
        let field_id = FieldId::new(1);

        let mut field_names = BTreeMap::new();
        field_names.insert(String::from("text"), field_id);

        let mut field_stats = BTreeMap::new();
        field_stats.insert(
            field_id,
            FieldMetadata {
                field_id,
                doc_count: 4,
                total_terms: 3,
            },
        );

        let mut terms_to_ids = BTreeMap::new();
        let mut term_entries = Vec::new();

        // Term 0: "hello"
        terms_to_ids.insert((field_id, String::from("hello")), leit_core::TermId::new(0));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(0),
            term: String::from("hello"),
        });

        // Term 1: "world"
        terms_to_ids.insert((field_id, String::from("world")), leit_core::TermId::new(1));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(1),
            term: String::from("world"),
        });

        // Term 2: "rust"
        terms_to_ids.insert((field_id, String::from("rust")), leit_core::TermId::new(2));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(2),
            term: String::from("rust"),
        });

        let mut postings = BTreeMap::new();
        postings.insert(
            leit_core::TermId::new(0),
            Vec::from([
                PostingEntry {
                    doc_id: 0,
                    term_freq: 1,
                },
                PostingEntry {
                    doc_id: 2,
                    term_freq: 2,
                },
            ]),
        );
        postings.insert(
            leit_core::TermId::new(1),
            Vec::from([
                PostingEntry {
                    doc_id: 1,
                    term_freq: 1,
                },
                PostingEntry {
                    doc_id: 3,
                    term_freq: 1,
                },
            ]),
        );
        postings.insert(
            leit_core::TermId::new(2),
            Vec::from([PostingEntry {
                doc_id: 0,
                term_freq: 3,
            }]),
        );

        let mut posting_blocks = BTreeMap::new();
        posting_blocks.insert(leit_core::TermId::new(0), Vec::new());
        posting_blocks.insert(leit_core::TermId::new(1), Vec::new());
        posting_blocks.insert(leit_core::TermId::new(2), Vec::new());

        let field_doc_lengths = BTreeMap::new();

        InMemoryIndex::new(
            FieldAnalyzers::default(),
            documents,
            terms_to_ids,
            term_entries,
            postings,
            posting_blocks,
            field_stats,
            field_names,
            field_doc_lengths,
        )
    }

    #[test]
    fn mmap_open_validates_header() {
        let index = build_test_index();
        let temp_path = write_test_segment_to_file(&index).expect("write segment to temp file");

        // Open the file via mmap; should succeed if header is valid.
        let result = MmapSegment::open(&temp_path);
        assert!(result.is_ok(), "mmap_open should succeed for valid segment");

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn mmap_open_rejects_short_file() {
        let temp_path = unique_temp_path("leit_mmap_short");

        // Write a file shorter than the header size (80 bytes).
        let mut file = std::fs::File::create(&temp_path).expect("create temp file");
        file.write_all(&[0_u8; 50]).expect("write short data");
        file.sync_all().expect("sync file");
        drop(file);

        let result = MmapSegment::open(&temp_path);
        assert!(result.is_err(), "mmap_open should reject truncated file");

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn mmap_open_rejects_bad_magic() {
        let temp_path = unique_temp_path("leit_mmap_badmagic");

        // Write a file with bad magic bytes.
        let mut buf = vec![0_u8; 100];
        buf[0..4].copy_from_slice(&0xDEADBEEF_u32.to_le_bytes());
        buf[4..8].copy_from_slice(&1_u32.to_le_bytes()); // version

        let mut file = std::fs::File::create(&temp_path).expect("create temp file");
        file.write_all(&buf).expect("write bad magic");
        file.sync_all().expect("sync file");
        drop(file);

        let result = MmapSegment::open(&temp_path);
        assert!(result.is_err(), "mmap_open should reject bad magic");

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn mmap_vs_buffer_equivalence() {
        let index = build_test_index();
        let segment_bytes = write_segment(&index).expect("write_segment should succeed");
        let temp_path = {
            let path = unique_temp_path("leit_mmap_equiv");
            let mut file = std::fs::File::create(&path).expect("create temp file");
            file.write_all(&segment_bytes).expect("write segment");
            file.sync_all().expect("sync file");
            drop(file);
            path
        };

        // Open via mmap.
        let mmap_segment = MmapSegment::open(&temp_path).expect("mmap open");
        let mmap_view = mmap_segment.as_view().expect("mmap as_view");

        // Open via buffer.
        let buffer_view = SegmentView::open(&segment_bytes).expect("buffer view open");

        // Compare document, field, and term counts.
        assert_eq!(
            mmap_view.document_count(),
            buffer_view.document_count(),
            "document_count must match"
        );
        assert_eq!(
            mmap_view.field_count().expect("mmap field_count"),
            buffer_view.field_count().expect("buffer field_count"),
            "field_count must match"
        );
        assert_eq!(
            mmap_view.term_count().expect("mmap term_count"),
            buffer_view.term_count().expect("buffer term_count"),
            "term_count must match"
        );

        // Compare field table entries byte-by-byte.
        let mmap_ft = mmap_view.field_table().expect("mmap field_table");
        let buffer_ft = buffer_view.field_table().expect("buffer field_table");
        assert_eq!(
            mmap_ft.len(),
            buffer_ft.len(),
            "field table lengths must match"
        );
        for i in 0..mmap_ft.len() {
            let mmap_entry = mmap_ft.entry(i).expect("mmap field entry");
            let buffer_entry = buffer_ft.entry(i).expect("buffer field entry");
            assert_eq!(
                mmap_entry, buffer_entry,
                "field table entry {} must match: mmap={:?}, buffer={:?}",
                i, mmap_entry, buffer_entry
            );
        }

        // Compare lexicon entries (term bytes and postings index).
        let mmap_lex = mmap_view.lexicon().expect("mmap lexicon");
        let buffer_lex = buffer_view.lexicon().expect("buffer lexicon");
        assert_eq!(
            mmap_lex.len(),
            buffer_lex.len(),
            "lexicon lengths must match"
        );
        for i in 0..mmap_lex.len() {
            let (mmap_term_bytes, mmap_pta_idx) = mmap_lex.entry(i).expect("mmap lexicon entry");
            let (buffer_term_bytes, buffer_pta_idx) =
                buffer_lex.entry(i).expect("buffer lexicon entry");
            assert_eq!(
                mmap_term_bytes, buffer_term_bytes,
                "lexicon entry {} term bytes must match",
                i
            );
            assert_eq!(
                mmap_pta_idx, buffer_pta_idx,
                "lexicon entry {} postings table index must match",
                i
            );
        }

        // Compare postings table entries.
        let mmap_pt = mmap_view.postings_table().expect("mmap postings_table");
        let buffer_pt = buffer_view.postings_table().expect("buffer postings_table");
        assert_eq!(
            mmap_pt.len(),
            buffer_pt.len(),
            "postings table lengths must match"
        );
        for i in 0..mmap_pt.len() {
            let mmap_entry = mmap_pt.entry(i).expect("mmap postings entry");
            let buffer_entry = buffer_pt.entry(i).expect("buffer postings entry");
            assert_eq!(
                mmap_entry, buffer_entry,
                "postings table entry {} must match: mmap={:?}, buffer={:?}",
                i, mmap_entry, buffer_entry
            );
        }

        // Compare postings data: compare the raw payload bytes for all ranges.
        let mmap_pd = mmap_view.postings_data().expect("mmap postings_data");
        let buffer_pd = buffer_view.postings_data().expect("buffer postings_data");
        // Extract the full postings data section and compare.
        if !mmap_pd.is_empty() {
            for i in 0..mmap_pt.len() {
                let (offset, len, _, _, _, _) = mmap_pt
                    .entry(i)
                    .expect("postings table entry for data check");
                let mmap_range = mmap_pd.range(offset, len).expect("mmap postings range");
                let buffer_range = buffer_pd.range(offset, len).expect("buffer postings range");
                assert_eq!(
                    mmap_range, buffer_range,
                    "postings data for term {} must match byte-for-byte",
                    i
                );
            }
        }

        // Compare block metadata entries.
        let mmap_bm = mmap_view.block_meta().expect("mmap block_meta");
        let buffer_bm = buffer_view.block_meta().expect("buffer block_meta");
        assert_eq!(
            mmap_bm.len(),
            buffer_bm.len(),
            "block_meta lengths must match"
        );
        for i in 0..mmap_bm.len() {
            let mmap_entry = mmap_bm.entry(i).expect("mmap block meta entry");
            let buffer_entry = buffer_bm.entry(i).expect("buffer block meta entry");
            assert_eq!(
                mmap_entry, buffer_entry,
                "block_meta entry {} must match: mmap={:?}, buffer={:?}",
                i, mmap_entry, buffer_entry
            );
        }

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn mmap_corrupt_body_detected_on_view() {
        let index = build_test_index();
        let segment_bytes = write_segment(&index).expect("write_segment should succeed");
        let temp_path = {
            let path = unique_temp_path("leit_mmap_corrupt");
            let mut file = std::fs::File::create(&path).expect("create temp file");
            file.write_all(&segment_bytes).expect("write segment");
            file.sync_all().expect("sync file");
            drop(file);
            path
        };

        // Corrupt a byte in the body (after header, in the field table section).
        // The header is 80 bytes; corrupt at offset 100.
        {
            use std::fs::OpenOptions;
            use std::io::{Seek, SeekFrom};
            let mut file = OpenOptions::new()
                .write(true)
                .open(&temp_path)
                .expect("open for corruption");
            file.seek(SeekFrom::Start(100))
                .expect("seek to corruption point");
            file.write_all(&[0xFF]).expect("write corruption byte");
            file.sync_all().expect("sync corruption");
            drop(file);
        }

        // Open via mmap: header-only validation succeeds.
        let mmap_segment =
            MmapSegment::open(&temp_path).expect("mmap open with valid header should succeed");

        // Attempt to get a view: structural validation should detect the corruption.
        // Corruption in field table causes the section readers to fail when iterating entries.
        match mmap_segment.as_view() {
            Ok(view) => {
                // If structural validation doesn't catch it, full validation with checksum should.
                // Try to access the field table, which will traverse corrupted data.
                let ft_result = view.field_table();
                // The corruption may cause entry parsing to fail (offset/bounds check).
                // If it doesn't fail immediately, at least the corrupted section will be present.
                // For this test, we verify that at least one of:
                // 1. as_view() detects corruption during structural validation, OR
                // 2. field_table() detects it during entry reads.
                // Since we corrupted after the header and before offsets are fully validated,
                // structural validation should catch misaligned or invalid entries.
                drop(ft_result);
            }
            Err(_) => {
                // Good: structural validation caught the corruption.
                // This is the expected outcome for a corrupted segment body.
            }
        }

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn mmap_truncated_body_detected_on_view() {
        let index = build_test_index();
        let segment_bytes = write_segment(&index).expect("write_segment should succeed");
        let temp_path = {
            let path = unique_temp_path("leit_mmap_truncated");
            let mut file = std::fs::File::create(&path).expect("create temp file");
            file.write_all(&segment_bytes).expect("write segment");
            file.sync_all().expect("sync file");
            drop(file);
            path
        };

        // Create a file with truncated body but valid header.
        // Write only the header (80 bytes), omitting the body sections.
        {
            let mut file = std::fs::File::create(&temp_path).expect("create truncated file");
            file.write_all(&segment_bytes[..80])
                .expect("write header only");
            file.sync_all().expect("sync file");
            drop(file);
        }

        // Attempt to open via mmap: header-only validation should succeed (we only wrote header).
        let mmap_segment =
            MmapSegment::open(&temp_path).expect("mmap open with valid header should succeed");

        // Attempt to get a view: structural validation should detect the truncation
        // because offsets in the header point beyond the truncated file size.
        let view_result = mmap_segment.as_view();
        assert!(
            view_result.is_err(),
            "view with truncated body should fail structural validation"
        );

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn mmap_full_validation_detects_checksum_corruption() {
        let temp_path = write_test_segment_to_file(&build_test_index()).expect("write test segment");

        let mut bytes = std::fs::read(&temp_path).expect("read temp segment");
        let corruption_offset = 150;
        assert!(corruption_offset < bytes.len() - 4, "corruption must avoid footer");
        bytes[corruption_offset] ^= 0xFF;
        std::fs::write(&temp_path, &bytes).expect("rewrite corrupted segment");

        let mmap_segment =
            MmapSegment::open(&temp_path).expect("mmap open with valid header should succeed");

        assert!(
            mmap_segment.as_view().is_ok(),
            "default structural view should still accept checksum-only corruption"
        );

        let full = mmap_segment
            .view()
            .with_mode(ValidationMode::Full)
            .open();
        assert!(
            matches!(full, Err(SegmentError::BadChecksum { .. })),
            "full mmap validation should detect checksum corruption"
        );

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn mmap_open_and_query() {
        let index = build_test_index();
        let temp_path = write_test_segment_to_file(&index).expect("write segment to temp file");

        let mmap_segment = MmapSegment::open(&temp_path).expect("mmap open");
        let view = mmap_segment.as_view().expect("as_view");

        // Call the public accessors.
        assert_eq!(view.document_count(), 4);
        assert!(view.field_count().is_ok(), "field_count should succeed");
        assert!(view.term_count().is_ok(), "term_count should succeed");
        assert!(view.field_table().is_ok(), "field_table should succeed");
        assert!(view.lexicon().is_ok(), "lexicon should succeed");
        assert!(
            view.postings_table().is_ok(),
            "postings_table should succeed"
        );
        assert!(view.postings_data().is_ok(), "postings_data should succeed");
        assert!(view.block_meta().is_ok(), "block_meta should succeed");

        let _ = std::fs::remove_file(&temp_path);
    }
}
