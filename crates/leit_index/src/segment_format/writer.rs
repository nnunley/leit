// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(dead_code, reason = "private encoder functions called internally")]

//! Segment writer: serializes `InMemoryIndex` to the DEC-05 v1 format.
//!
//! This module implements the writer for the Phase 2 segment format (DEC-05), producing:
//! - Fixed 80-byte `SegmentHeader` at offset 0
//! - 4 populated sections: `field_table`, `lexicon`, `postings_table`, `postings_data`
//! - 3 reserved zero-length sections: `block_meta`, `stored_fields`, `columnar`
//! - Footer with CRC32C checksum at `footer_offset`
//!
//! **Builder/Reader Separation (DEC-09, STORY-0084 AC-2):**
//! The writer consumes borrowed pieces of `InMemoryIndex` and emits bytes. It does NOT expose
//! builder types in the public API; all writer internals are private. The output conforms exactly
//! to the T4 reader layouts for round-trip compatibility.
//!
//! **Format Flags (DEC-10):**
//! - Bit 0: `optional_sections_present` = 0 (`block_meta`, `stored_fields`, `columnar` are absent/zero-length)
//! - Bits 1-31: reserved for future use
//!
//! **Codec ID Marker (STORY-0002 AC-3):**
//! The postings table reserves space for codec selection per-term (ITER-0005). For v1-core,
//! codec selection is not stored; all postings use uncompressed format.

use alloc::vec::Vec;

use crate::error::IndexError;
use crate::memory::InMemoryIndex;
use crate::segment_format::footer::{Footer, compute_checksum};
use crate::segment_format::header::{FORMAT_VERSION, HEADER_SIZE, MAGIC, SegmentHeader};

/// Format flags: indicate which optional sections are present.
/// In v1-core (ITER-0004), all optional sections are zero-length, so this is always 0.
const FORMAT_FLAGS_V1_CORE: u32 = 0;

/// Reserved codec ID for uncompressed postings (STORY-0002 AC-3).
/// Not used in v1-core; reserved for ITER-0005 when postings codecs are implemented.
const RESERVED_CODEC_ID_UNCOMPRESSED: u32 = 0;

/// Serialize an `InMemoryIndex` to the DEC-05 v1 segment format.
///
/// This is the main entry point. The function:
/// 1. Encodes field table, lexicon, postings table, and postings data sections.
/// 2. Computes absolute offsets for each section.
/// 3. Reserves zero-length slots for `block_meta`, `stored_fields`, `columnar`.
/// 4. Writes the `SegmentHeader` at offset 0.
/// 5. Writes the `Footer` at `footer_offset` with a CRC32C checksum.
///
/// # Returns
/// A `Vec<u8>` containing the complete segment buffer.
pub(crate) fn write_segment(index: &InMemoryIndex) -> Result<Vec<u8>, IndexError> {
    // Phase 1: Encode sections to byte vectors (sizes not yet known).
    let field_table = encode_field_table(index)?;
    let lexicon = encode_lexicon(index)?;
    let (postings_table, postings_data) = encode_postings(index)?;

    // Phase 2: Compute absolute offsets.
    let mut offset = HEADER_SIZE as u64;

    let field_table_offset = offset;
    offset = offset
        .checked_add(field_table.len() as u64)
        .ok_or(IndexError::ValueOutOfRange)?;

    let lexicon_offset = offset;
    offset = offset
        .checked_add(lexicon.len() as u64)
        .ok_or(IndexError::ValueOutOfRange)?;

    let postings_table_offset = offset;
    offset = offset
        .checked_add(postings_table.len() as u64)
        .ok_or(IndexError::ValueOutOfRange)?;

    let postings_data_offset = offset;
    offset = offset
        .checked_add(postings_data.len() as u64)
        .ok_or(IndexError::ValueOutOfRange)?;

    // Reserved sections (zero-length in v1-core).
    let block_meta_offset = offset;
    let stored_fields_offset = offset;
    let columnar_offset = offset;

    let footer_offset = offset;

    // Footer is 4 bytes (note: offset is not used after this, but we validate it for completeness).
    let _footer_end = offset.checked_add(4).ok_or(IndexError::ValueOutOfRange)?;

    // Phase 3: Assemble the segment.
    let mut segment = Vec::new();

    // Write header at offset 0.
    let header = SegmentHeader {
        magic: MAGIC,
        version: FORMAT_VERSION,
        format_flags: FORMAT_FLAGS_V1_CORE,
        reserved: 0,
        field_table_offset,
        lexicon_offset,
        postings_table_offset,
        postings_data_offset,
        block_meta_offset,
        stored_fields_offset,
        columnar_offset,
        footer_offset,
    };
    segment.extend_from_slice(&header.encode());

    // Write sections.
    segment.extend_from_slice(&field_table);
    segment.extend_from_slice(&lexicon);
    segment.extend_from_slice(&postings_table);
    segment.extend_from_slice(&postings_data);

    // Write footer.
    let footer_offset_usize =
        usize::try_from(footer_offset).map_err(|_| IndexError::ValueOutOfRange)?;
    let checksum = compute_checksum(&segment[..footer_offset_usize]);
    let footer = Footer { checksum };
    segment.extend_from_slice(&footer.encode());

    Ok(segment)
}

/// Encode the field table section.
///
/// Layout (per `FieldTableReader`):
/// - Offset 0: count (`u32` LE)
/// - Offset 4..: entries (12 bytes each): `field_id` (`u32`) + `doc_count` (`u32`) + `total_terms` (`u32`)
fn encode_field_table(index: &InMemoryIndex) -> Result<Vec<u8>, IndexError> {
    let mut bytes = Vec::new();

    let count =
        u32::try_from(index.field_stats().len()).map_err(|_| IndexError::ValueOutOfRange)?;
    push_u32(&mut bytes, count);

    for stats in index.field_stats().values() {
        push_u32(&mut bytes, stats.field_id.as_u32());
        push_u32(&mut bytes, stats.doc_count);
        push_u32(&mut bytes, stats.total_terms);
    }

    Ok(bytes)
}

/// Encode the lexicon section.
///
/// Layout (per `LexiconReader`):
/// - Offset 0: count (`u32` LE)
/// - Offset 4..: index entries (16 bytes each):
///   - `term_offset` (`u64`): offset relative to blob start (computed below)
///   - `term_len` (`u32`): length of term bytes
///   - `postings_table_index` (`u32`): index into `postings_table` for this term's postings
/// - After index: variable-length term bytes blob
fn encode_lexicon(index: &InMemoryIndex) -> Result<Vec<u8>, IndexError> {
    let mut bytes = Vec::new();
    let entries = index.term_entries();

    let count = u32::try_from(entries.len()).map_err(|_| IndexError::ValueOutOfRange)?;
    push_u32(&mut bytes, count);

    // Reserve space for index entries (16 bytes each).
    let index_size = count as usize * 16;
    bytes.resize(bytes.len() + index_size, 0);

    // Collect term bytes to compute offsets.
    let mut term_bytes_blob = Vec::new();
    let mut index_offset = 4_usize; // Offset into `bytes` where index entries start.

    for (idx, entry) in entries.iter().enumerate() {
        let term_raw = entry.term.as_bytes();
        let term_offset =
            u64::try_from(term_bytes_blob.len()).map_err(|_| IndexError::ValueOutOfRange)?;
        let term_len = u32::try_from(term_raw.len()).map_err(|_| IndexError::ValueOutOfRange)?;

        // postings_table_index: simply the index of this term in the term_entries array.
        // In the new format, we map term indices to postings table indices (which are the same).
        let postings_table_index = u32::try_from(idx).map_err(|_| IndexError::ValueOutOfRange)?;

        // Write index entry: term_offset (u64) + term_len (u32) + postings_table_index (u32).
        push_u64_at(&mut bytes, index_offset, term_offset);
        push_u32_at(&mut bytes, index_offset + 8, term_len);
        push_u32_at(&mut bytes, index_offset + 12, postings_table_index);
        index_offset += 16;

        // Append term bytes to blob.
        term_bytes_blob.extend_from_slice(term_raw);
    }

    // Append term bytes blob to the end.
    bytes.extend_from_slice(&term_bytes_blob);

    Ok(bytes)
}

/// Encode the postings table and postings data sections.
///
/// Returns (`postings_table`, `postings_data`) as separate byte vectors.
///
/// **IMPORTANT: Iteration Order**
/// The postings table entries MUST be in the same order as the lexicon entries.
/// Both iterate in the order of `index.term_entries()` (by `TermId`).
///
/// **Postings Table Layout (per `PostingsTableReader`, STORY-0002 AC-3):**
/// - Offset 0: count (`u32` LE)
/// - Offset 4..: entries (20 bytes each):
///   - `postings_data_offset` (`u64`, offset 0): absolute offset within `postings_data` section
///   - `postings_data_len` (`u32`, offset 8): byte length of `postings_data` for this term
///   - `doc_freq` (`u32`, offset 12): number of documents containing this term (postings count)
///   - `reserved_codec_id` (`u32`, offset 16): reserved for codec selection; 0 for v1-core
///
/// **Postings Data Layout (per `PostingsDataReader`):**
/// Raw bytes concatenated from all terms. Each term's postings are encoded as:
/// - Postings for a term are stored as delta-encoded doc IDs followed by term frequencies.
///   For v1-core, we use a simple uncompressed format: each posting is (`doc_id` `u32` + `term_freq` `u32`).
fn encode_postings(index: &InMemoryIndex) -> Result<(Vec<u8>, Vec<u8>), IndexError> {
    let mut table = Vec::new();
    let mut data = Vec::new();

    let entries = index.term_entries();
    let postings_map = index.postings();
    let count = u32::try_from(entries.len()).map_err(|_| IndexError::ValueOutOfRange)?;

    push_u32(&mut table, count);

    // Iterate in the same order as term_entries to maintain correspondence with lexicon.
    for entry in entries {
        let Some(postings) = postings_map.get(&entry.term_id) else {
            return Err(IndexError::ValueOutOfRange); // Term exists in lexicon but not in postings
        };
        let data_offset = u64::try_from(data.len()).map_err(|_| IndexError::ValueOutOfRange)?;

        // Encode postings: each posting as (doc_id u32 + term_freq u32).
        for posting in postings {
            push_u32(&mut data, posting.doc_id);
            push_u32(&mut data, posting.term_freq);
        }

        let data_offset_usize =
            usize::try_from(data_offset).map_err(|_| IndexError::ValueOutOfRange)?;
        let data_offset_u32 =
            u32::try_from(data_offset_usize).map_err(|_| IndexError::ValueOutOfRange)?;
        let data_len = u32::try_from(data.len())
            .map_err(|_| IndexError::ValueOutOfRange)?
            .checked_sub(data_offset_u32)
            .ok_or(IndexError::ValueOutOfRange)?;

        let doc_freq = u32::try_from(postings.len()).map_err(|_| IndexError::ValueOutOfRange)?;

        // Write postings table entry (20 bytes): offset (u64), len (u32), freq (u32), codec_id (u32).
        // STORY-0002 AC-3: reserve codec_id field for per-term codec selection (ITER-0005).
        push_u64(&mut table, data_offset);
        push_u32(&mut table, data_len);
        push_u32(&mut table, doc_freq);
        push_u32(&mut table, RESERVED_CODEC_ID_UNCOMPRESSED);
    }

    Ok((table, data))
}

/// Write a u32 LE to a byte vector.
#[inline]
fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

/// Write a u64 LE to a byte vector.
#[inline]
fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

/// Write a u32 LE at a specific offset in a byte slice.
#[inline]
fn push_u32_at(bytes: &mut [u8], offset: usize, value: u32) {
    let value_bytes = value.to_le_bytes();
    bytes[offset..offset + 4].copy_from_slice(&value_bytes);
}

/// Write a u64 LE at a specific offset in a byte slice.
#[inline]
fn push_u64_at(bytes: &mut [u8], offset: usize, value: u64) {
    let value_bytes = value.to_le_bytes();
    bytes[offset..offset + 8].copy_from_slice(&value_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::{BTreeMap, BTreeSet};
    use alloc::string::String;
    use alloc::vec;

    use leit_core::FieldId;
    use leit_text::FieldAnalyzers;

    use crate::memory::{FieldMetadata, PostingEntry, TermEntry};
    use crate::segment_format::readers::{
        FieldTableReader, LexiconReader, PostingsDataReader, PostingsTableReader,
    };

    /// Build a minimal test `InMemoryIndex` with known data.
    fn make_test_index() -> InMemoryIndex {
        // 2 documents
        let documents = BTreeSet::from([0, 1]);

        // 1 field
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

        // 3 terms
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

        // Term "rust" (term_id = 2)
        terms_to_ids.insert((field_id, String::from("rust")), leit_core::TermId::new(2));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(2),
            term: String::from("rust"),
        });

        // Postings:
        // Term 0 "hello": doc 0 (freq 1), doc 1 (freq 1)
        // Term 1 "world": doc 0 (freq 2)
        // Term 2 "rust": doc 1 (freq 3)
        let mut postings = BTreeMap::new();
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
                term_freq: 2,
            }],
        );
        postings.insert(
            leit_core::TermId::new(2),
            vec![PostingEntry {
                doc_id: 1,
                term_freq: 3,
            }],
        );

        let mut posting_blocks = BTreeMap::new();
        posting_blocks.insert(leit_core::TermId::new(0), vec![]);
        posting_blocks.insert(leit_core::TermId::new(1), vec![]);
        posting_blocks.insert(leit_core::TermId::new(2), vec![]);

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
    fn test_roundtrip_populated_sections() {
        let index = make_test_index();

        // Debug: trace section encoding (sections must be non-empty for assertions to make sense)
        let ft = encode_field_table(&index).expect("ft encode");
        let lex = encode_lexicon(&index).expect("lex encode");
        let (pt, _pd) = encode_postings(&index).expect("postings encode");

        // Verify sections are non-empty
        assert!(ft.len() >= 4);
        assert!(lex.len() >= 4);
        assert!(pt.len() >= 4);

        // Write the segment.
        let segment = write_segment(&index).expect("write_segment should succeed");

        // Verify basic structure: header + sections + footer
        assert!(
            segment.len() > HEADER_SIZE + 4,
            "segment must be larger than header + footer"
        );

        // Read header.
        let header = SegmentHeader::read(&segment).expect("header should decode");
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, FORMAT_VERSION);
        assert_eq!(header.format_flags, FORMAT_FLAGS_V1_CORE);

        // Verify offset ordering (all should be >= HEADER_SIZE and monotonic).
        assert_eq!(header.field_table_offset, HEADER_SIZE as u64);
        assert!(header.lexicon_offset > header.field_table_offset);
        assert!(header.postings_table_offset > header.lexicon_offset);
        assert!(header.postings_data_offset > header.postings_table_offset);
        // In v1-core, block_meta, stored_fields, columnar are all zero-length (same offset).
        assert_eq!(
            header.block_meta_offset, header.stored_fields_offset,
            "block_meta and stored_fields should have same offset (both zero-length)"
        );
        assert_eq!(
            header.columnar_offset, header.block_meta_offset,
            "columnar should have same offset as block_meta (zero-length)"
        );
        // footer comes after all sections
        assert!(header.footer_offset >= header.columnar_offset);

        // Read field table.
        let ft = FieldTableReader::new(&segment, header.field_table_offset, header.lexicon_offset)
            .expect("FieldTableReader should construct");
        assert_eq!(ft.len(), 1, "should have 1 field");
        let (field_id, doc_count, total_terms) = ft.entry(0).expect("field 0 should exist");
        assert_eq!(field_id, 1); // FieldId::new(1).as_u32() == 1
        assert_eq!(doc_count, 2);
        assert_eq!(total_terms, 3);

        // Read lexicon.
        let lex = LexiconReader::new(
            &segment,
            header.lexicon_offset,
            header.postings_table_offset,
        )
        .expect("LexiconReader should construct");
        assert_eq!(lex.len(), 3, "should have 3 terms");

        // Verify term 0 "hello".
        let (term_bytes, ptindex) = lex.entry(0).expect("term 0 should exist");
        assert_eq!(term_bytes, b"hello");
        assert_eq!(ptindex, 0, "term 0 should map to postings table index 0");

        // Verify term 1 "world".
        let (term_bytes, ptindex) = lex.entry(1).expect("term 1 should exist");
        assert_eq!(term_bytes, b"world");
        assert_eq!(
            ptindex, 1,
            "lexicon entry 1 should point to postings table index 1"
        );

        // Verify term 2 "rust".
        let (term_bytes, ptindex) = lex.entry(2).expect("term 2 should exist");
        assert_eq!(term_bytes, b"rust");
        assert_eq!(ptindex, 2);

        // Read postings table.
        let pt = PostingsTableReader::new(
            &segment,
            header.postings_table_offset,
            header.postings_data_offset,
        )
        .expect("PostingsTableReader should construct");
        assert_eq!(pt.len(), 3, "should have 3 postings entries");

        // Postings table entry 0: term "hello" with 2 postings (doc 0, doc 1).
        let (pdata_offset, pdata_len, doc_freq, codec_id) =
            pt.entry(0).expect("postings entry 0 should exist");
        // Each posting is 8 bytes; "hello" has 2 postings.
        assert_eq!(pdata_len, 16, "term 'hello' data_len should be 16 bytes");
        assert_eq!(doc_freq, 2, "term 'hello' has doc_freq=2");
        assert_eq!(codec_id, 0, "reserved_codec_id should be 0 for v1-core");

        // Postings table entry 1: term "world" with 1 posting (doc 0).
        let (_pdata_offset1, _pdata_len1, doc_freq1, codec_id1) =
            pt.entry(1).expect("postings entry 1 should exist");
        assert_eq!(doc_freq1, 1, "term 'world' has doc_freq=1");
        assert_eq!(codec_id1, 0, "reserved_codec_id should be 0 for v1-core");

        // Postings table entry 2: term "rust" with 1 posting (doc 1).
        let (_pdata_offset2, _pdata_len2, doc_freq2, codec_id2) =
            pt.entry(2).expect("postings entry 2 should exist");
        assert_eq!(doc_freq2, 1, "term 'rust' has doc_freq=1");
        assert_eq!(codec_id2, 0, "reserved_codec_id should be 0 for v1-core");

        // Read postings data.
        let pd = PostingsDataReader::new(
            &segment,
            header.postings_data_offset,
            header.block_meta_offset,
        )
        .expect("PostingsDataReader should construct");

        // Verify postings data for term 0 "hello": (doc_id=0, freq=1), (doc_id=1, freq=1).
        let hello_data = pd
            .range(pdata_offset, pdata_len)
            .expect("postings range 0 should exist");
        assert_eq!(hello_data.len(), pdata_len as usize);
        // Each posting is 8 bytes (doc_id u32 + term_freq u32). Term "hello" has 2 postings = 16 bytes.
        assert_eq!(hello_data.len(), 16);
        let doc_id_0 =
            u32::from_le_bytes([hello_data[0], hello_data[1], hello_data[2], hello_data[3]]);
        let freq_0 =
            u32::from_le_bytes([hello_data[4], hello_data[5], hello_data[6], hello_data[7]]);
        assert_eq!(doc_id_0, 0);
        assert_eq!(freq_0, 1);

        let doc_id_1 =
            u32::from_le_bytes([hello_data[8], hello_data[9], hello_data[10], hello_data[11]]);
        let freq_1 = u32::from_le_bytes([
            hello_data[12],
            hello_data[13],
            hello_data[14],
            hello_data[15],
        ]);
        assert_eq!(doc_id_1, 1);
        assert_eq!(freq_1, 1);

        // Verify footer and checksum.
        let footer_offset_usize = usize::try_from(header.footer_offset).unwrap();
        let footer_bytes = &segment[footer_offset_usize..footer_offset_usize + 4];
        let footer_checksum = u32::from_le_bytes([
            footer_bytes[0],
            footer_bytes[1],
            footer_bytes[2],
            footer_bytes[3],
        ]);

        // Recompute checksum.
        let computed_checksum = compute_checksum(&segment[..footer_offset_usize]);
        assert_eq!(
            footer_checksum, computed_checksum,
            "footer checksum should match"
        );
    }

    /// Test STORY-0002 AC-3: `postings_table` entries are 20 bytes with reserved `codec_id` slot.
    #[test]
    fn test_postings_table_entry_is_20_bytes_with_codec_id_slot() {
        let index = make_test_index();
        let segment = write_segment(&index).expect("write_segment should succeed");
        let header = SegmentHeader::read(&segment).expect("header should decode");

        // Read postings table and verify all entries round-trip with codec_id.
        let pt = PostingsTableReader::new(
            &segment,
            header.postings_table_offset,
            header.postings_data_offset,
        )
        .expect("PostingsTableReader should construct");
        assert_eq!(pt.len(), 3, "should have 3 postings entries");

        // Each postings table entry must be 20 bytes (4-byte count + 3 * 20-byte entries).
        let pt_section_size =
            usize::try_from(header.postings_data_offset - header.postings_table_offset)
                .unwrap_or(0);
        let expected_pt_size = 4 + (3 * 20); // count + 3 entries
        assert_eq!(
            pt_section_size, expected_pt_size,
            "postings table section must be 4 (count) + 3 * 20 (entries) = 64 bytes"
        );

        // Verify all entries have codec_id = 0 (RESERVED_CODEC_ID_UNCOMPRESSED).
        for i in 0..3 {
            let (_offset, _len, _doc_freq, codec_id) = pt.entry(i).expect("entry should exist");
            assert_eq!(
                codec_id, 0,
                "entry {} reserved_codec_id must be 0 (uncompressed marker)",
                i
            );
        }

        // Verify the codec_id slot is actually read from the segment bytes
        // by checking that directly reading from the segment gives the same value.
        let pt_start = usize::try_from(header.postings_table_offset).unwrap_or(0);
        // First entry starts at pt_start + 4 (count)
        // codec_id is at offset 16 within each entry
        let first_entry_codec_id_offset = pt_start + 4 + 16;
        let codec_id_bytes = &segment[first_entry_codec_id_offset..first_entry_codec_id_offset + 4];
        let codec_id_from_bytes = u32::from_le_bytes([
            codec_id_bytes[0],
            codec_id_bytes[1],
            codec_id_bytes[2],
            codec_id_bytes[3],
        ]);
        assert_eq!(
            codec_id_from_bytes, 0,
            "reserved_codec_id slot must exist at offset 16 in segment bytes"
        );
    }
}
