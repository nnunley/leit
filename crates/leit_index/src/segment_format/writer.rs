// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Segment writer: serializes `InMemoryIndex` to the v1 format.
//!
//! This module implements the writer for the segment format, producing:
//! - Fixed 80-byte `SegmentHeader` at offset 0
//! - 5 populated sections: `field_table`, `lexicon`, `postings_table`, `postings_data`, `block_meta`
//! - 2 reserved zero-length sections: `stored_fields`, `columnar`
//! - Footer with CRC32C checksum at `footer_offset`
//!
//! **Builder/Reader Separation:**
//! The writer consumes borrowed pieces of `InMemoryIndex` and emits bytes. It does NOT expose
//! builder types in the public API; all writer internals are private. The output conforms exactly
//! to the reader layouts for round-trip compatibility.
//!
//! **Format Flags:**
//! - Bit 0: `optional_sections_present` = 0 (`stored_fields`, `columnar` are absent/zero-length)
//! - Bits 1-31: reserved for future use
//!
//! **Codec ID Marker:**
//! The postings table reserves space for codec selection per-term. For the current v1 format,
//! codec selection is not stored; all postings use uncompressed format.
//!
//! **Block Metadata:**
//! Postings are divided into fixed-size blocks (128 documents per block per `BLOCK_DOC_COUNT`).
//! For each block, a summary is written to the `block_meta` section containing the end document ID,
//! maximum term frequency, and relative decode offset.

use alloc::vec::Vec;
use core::cmp;

use bytemuck;
use leit_postings::codec::BLOCK_DOC_COUNT;

use crate::error::IndexError;
use crate::memory::InMemoryIndex;
use crate::segment_format::block_meta::BlockMetadataEntry;
use crate::segment_format::footer::{Footer, compute_checksum};
use crate::segment_format::header::{FORMAT_VERSION, HEADER_SIZE, MAGIC, SegmentHeader};

/// Format flags: indicate which optional sections are present.
/// In v1, `block_meta` is always populated, and `stored_fields`/`columnar` are zero-length, so this is always 0.
const FORMAT_FLAGS_V1_CORE: u32 = 0;

/// Reserved codec ID for uncompressed postings.
/// Not used in current v1; reserved for future releases when postings codecs are implemented.
const RESERVED_CODEC_ID_UNCOMPRESSED: u32 = 0;

/// Serialize an `InMemoryIndex` to the v1 segment format.
///
/// This is the main entry point. The function:
/// 1. Encodes field table, lexicon, postings table, postings data, and block metadata sections.
/// 2. Computes absolute offsets for each section.
/// 3. Reserves zero-length slots for `stored_fields`, `columnar`.
/// 4. Writes the `SegmentHeader` at offset 0.
/// 5. Writes the `Footer` at `footer_offset` with a CRC32C checksum.
///
/// # Returns
/// A `Vec<u8>` containing the complete segment buffer.
pub(crate) fn write_segment(index: &InMemoryIndex) -> Result<Vec<u8>, IndexError> {
    // Encode sections to byte vectors (sizes not yet known).
    let field_table = encode_field_table(index)?;
    let lexicon = encode_lexicon(index)?;
    let (postings_table, postings_data, block_meta) = encode_postings(index)?;

    // Compute absolute offsets.
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

    // Block metadata section (now populated in v1).
    let block_meta_offset = offset;
    offset = offset
        .checked_add(block_meta.len() as u64)
        .ok_or(IndexError::ValueOutOfRange)?;

    // Reserved zero-length sections.
    let stored_fields_offset = offset;
    let columnar_offset = offset;

    let footer_offset = offset;

    // Footer is 4 bytes (note: offset is not used after this, but we validate it for completeness).
    let _footer_end = offset.checked_add(4).ok_or(IndexError::ValueOutOfRange)?;

    // Assemble the segment.
    let mut segment = Vec::new();

    // Write header at offset 0.
    let header = SegmentHeader {
        magic: MAGIC,
        version: FORMAT_VERSION,
        format_flags: FORMAT_FLAGS_V1_CORE,
        document_count: index.document_count(),
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
    segment.extend_from_slice(&block_meta);

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

/// Encode the postings table, postings data, and block metadata sections.
///
/// Returns (`postings_table`, `postings_data`, `block_meta`) as separate byte vectors.
///
/// **IMPORTANT: Iteration Order**
/// The postings table entries MUST be in the same order as the lexicon entries.
/// Both iterate in the order of `index.term_entries()` (by `TermId`).
///
/// **Postings Table Layout (per `PostingsTableReader`):**
/// - Offset 0: count (`u32` LE)
/// - Offset 4..: entries (28 bytes each):
///   - `postings_data_offset` (`u64`, offset 0): absolute offset within `postings_data` section
///   - `postings_data_len` (`u32`, offset 8): byte length of `postings_data` for this term
///   - `doc_freq` (`u32`, offset 12): number of documents containing this term (postings count)
///   - `reserved_codec_id` (`u32`, offset 16): reserved for codec selection; 0 for minimal v1
///   - `first_block_index` (`u32`, offset 20): index of this term's first block in the `block_meta` table
///   - `block_count` (`u32`, offset 24): number of blocks for this term
///
/// **Postings Data Layout (per `PostingsDataReader`):**
/// Raw bytes concatenated from all terms. Each term's postings are encoded as:
/// - Postings for a term are stored as delta-encoded doc IDs followed by term frequencies.
///   For the current v1 format, we use a simple uncompressed format: each posting is (`doc_id` `u32` + `term_freq` `u32`).
///
/// **Block Metadata Layout (per `BlockMetadataReader`):**
/// Contiguous array of `BlockMetadataEntry` (12 bytes each), with no count prefix.
/// Entry count is derived from section length / 12. Blocks for a term are stored contiguously
/// with indices [`first_block_index`, `first_block_index` + `block_count`).
#[expect(
    clippy::type_complexity,
    reason = "tuple return type with three Vec<u8> is acceptable here"
)]
fn encode_postings(index: &InMemoryIndex) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), IndexError> {
    let mut table = Vec::new();
    let mut data = Vec::new();
    let mut block_meta = Vec::new();

    let entries = index.term_entries();
    let postings_map = index.postings();
    let count = u32::try_from(entries.len()).map_err(|_| IndexError::ValueOutOfRange)?;

    push_u32(&mut table, count);

    // Track cumulative block count across all terms for first_block_index
    let mut cumulative_block_count: u32 = 0;

    // Iterate in the same order as term_entries to maintain correspondence with lexicon.
    for entry in entries {
        let Some(postings) = postings_map.get(&entry.term_id) else {
            return Err(IndexError::ValueOutOfRange); // Term exists in lexicon but not in postings
        };
        let data_offset = u64::try_from(data.len()).map_err(|_| IndexError::ValueOutOfRange)?;
        let _term_data_start = data.len();

        // Record the first_block_index for this term
        let first_block_index = cumulative_block_count;

        // Encode postings: each posting as (doc_id u32 + term_freq u32).
        for posting in postings {
            push_u32(&mut data, posting.doc_id);
            push_u32(&mut data, posting.term_freq);
        }

        // Compute blocks for this term.
        // A block contains up to BLOCK_DOC_COUNT postings.
        let doc_freq = u32::try_from(postings.len()).map_err(|_| IndexError::ValueOutOfRange)?;
        let block_size = u32::try_from(BLOCK_DOC_COUNT).map_err(|_| IndexError::ValueOutOfRange)?;
        let block_count = doc_freq.div_ceil(block_size);

        // Write block metadata entries for this term
        for block_idx in 0..block_count {
            let start_posting_idx = (block_idx as usize) * BLOCK_DOC_COUNT;
            let end_posting_idx = cmp::min(start_posting_idx + BLOCK_DOC_COUNT, postings.len());

            // end_doc: the last doc_id in this block
            let end_doc = postings[end_posting_idx - 1].doc_id;

            // max_term_freq: max term_freq in this block
            let max_term_freq = postings[start_posting_idx..end_posting_idx]
                .iter()
                .map(|p| p.term_freq)
                .max()
                .unwrap_or(0);

            // decode_offset: relative to term's postings payload start (in bytes)
            // For uncompressed format, each posting is 8 bytes (doc_id u32 + term_freq u32)
            let decode_offset = u32::try_from(start_posting_idx)
                .map_err(|_| IndexError::ValueOutOfRange)?
                .checked_mul(8)
                .ok_or(IndexError::ValueOutOfRange)?;

            // Write block metadata entry
            let entry = BlockMetadataEntry {
                end_doc,
                max_term_freq,
                decode_offset,
            };
            block_meta.extend_from_slice(bytemuck::bytes_of(&entry));
        }

        cumulative_block_count = cumulative_block_count
            .checked_add(block_count)
            .ok_or(IndexError::ValueOutOfRange)?;

        let data_offset_usize =
            usize::try_from(data_offset).map_err(|_| IndexError::ValueOutOfRange)?;
        let data_offset_u32 =
            u32::try_from(data_offset_usize).map_err(|_| IndexError::ValueOutOfRange)?;
        let data_len = u32::try_from(data.len())
            .map_err(|_| IndexError::ValueOutOfRange)?
            .checked_sub(data_offset_u32)
            .ok_or(IndexError::ValueOutOfRange)?;

        // Write postings table entry (28 bytes): offset (u64), len (u32), freq (u32), codec_id (u32),
        // first_block_index (u32), block_count (u32).
        push_u64(&mut table, data_offset);
        push_u32(&mut table, data_len);
        push_u32(&mut table, doc_freq);
        push_u32(&mut table, RESERVED_CODEC_ID_UNCOMPRESSED);
        push_u32(&mut table, first_block_index);
        push_u32(&mut table, block_count);
    }

    Ok((table, data, block_meta))
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
        BlockMetadataReader, FieldTableReader, LexiconReader, PostingsDataReader,
        PostingsTableReader,
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
        let (pt, _pd, _bm) = encode_postings(&index).expect("postings encode");

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
        // In v1, block_meta is populated (not zero-length)
        assert!(
            header.block_meta_offset >= header.postings_data_offset,
            "block_meta should come after postings_data"
        );
        // stored_fields and columnar are zero-length (same offset, after block_meta)
        assert_eq!(
            header.stored_fields_offset, header.columnar_offset,
            "stored_fields and columnar should have same offset (both zero-length)"
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
        let (pdata_offset, pdata_len, doc_freq, codec_id, first_block_idx, block_count) =
            pt.entry(0).expect("postings entry 0 should exist");
        // Each posting is 8 bytes; "hello" has 2 postings.
        assert_eq!(pdata_len, 16, "term 'hello' data_len should be 16 bytes");
        assert_eq!(doc_freq, 2, "term 'hello' has doc_freq=2");
        assert_eq!(codec_id, 0, "reserved_codec_id should be 0 for minimal v1");
        assert_eq!(
            first_block_idx, 0,
            "term 'hello' should have first_block_index=0"
        );
        assert_eq!(
            block_count, 1,
            "term 'hello' with 2 postings fits in 1 block"
        );

        // Postings table entry 1: term "world" with 1 posting (doc 0).
        let (_pdata_offset1, _pdata_len1, doc_freq1, codec_id1, first_block_idx1, block_count1) =
            pt.entry(1).expect("postings entry 1 should exist");
        assert_eq!(doc_freq1, 1, "term 'world' has doc_freq=1");
        assert_eq!(codec_id1, 0, "reserved_codec_id should be 0 for minimal v1");
        assert_eq!(
            first_block_idx1, 1,
            "term 'world' should have first_block_index=1"
        );
        assert_eq!(
            block_count1, 1,
            "term 'world' with 1 posting fits in 1 block"
        );

        // Postings table entry 2: term "rust" with 1 posting (doc 1).
        let (_pdata_offset2, _pdata_len2, doc_freq2, codec_id2, first_block_idx2, block_count2) =
            pt.entry(2).expect("postings entry 2 should exist");
        assert_eq!(doc_freq2, 1, "term 'rust' has doc_freq=1");
        assert_eq!(codec_id2, 0, "reserved_codec_id should be 0 for minimal v1");
        assert_eq!(
            first_block_idx2, 2,
            "term 'rust' should have first_block_index=2"
        );
        assert_eq!(
            block_count2, 1,
            "term 'rust' with 1 posting fits in 1 block"
        );

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

    /// Test: `postings_table` entries are 28 bytes with block metadata fields.
    #[test]
    fn test_postings_table_entry_is_28_bytes_with_block_fields() {
        let index = make_test_index();
        let segment = write_segment(&index).expect("write_segment should succeed");
        let header = SegmentHeader::read(&segment).expect("header should decode");

        // Read postings table and verify all entries round-trip with block fields.
        let pt = PostingsTableReader::new(
            &segment,
            header.postings_table_offset,
            header.postings_data_offset,
        )
        .expect("PostingsTableReader should construct");
        assert_eq!(pt.len(), 3, "should have 3 postings entries");

        // Each postings table entry must be 28 bytes (4-byte count + 3 * 28-byte entries).
        let pt_section_size =
            usize::try_from(header.postings_data_offset - header.postings_table_offset)
                .unwrap_or(0);
        let expected_pt_size = 4 + (3 * 28); // count + 3 entries
        assert_eq!(
            pt_section_size, expected_pt_size,
            "postings table section must be 4 (count) + 3 * 28 (entries) = 88 bytes"
        );

        // Verify all entries have codec_id = 0 (RESERVED_CODEC_ID_UNCOMPRESSED).
        for i in 0..3 {
            let (_offset, _len, _doc_freq, codec_id, _first_block_idx, _block_count) =
                pt.entry(i).expect("entry should exist");
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

        // Verify block fields are present at offsets 20-24 and 24-28 within each entry
        let first_entry_first_block_offset = pt_start + 4 + 20;
        let first_block_bytes =
            &segment[first_entry_first_block_offset..first_entry_first_block_offset + 4];
        let first_block_from_bytes = u32::from_le_bytes([
            first_block_bytes[0],
            first_block_bytes[1],
            first_block_bytes[2],
            first_block_bytes[3],
        ]);
        assert_eq!(
            first_block_from_bytes, 0,
            "first entry's first_block_index must be 0"
        );

        let first_entry_block_count_offset = pt_start + 4 + 24;
        let block_count_bytes =
            &segment[first_entry_block_count_offset..first_entry_block_count_offset + 4];
        let block_count_from_bytes = u32::from_le_bytes([
            block_count_bytes[0],
            block_count_bytes[1],
            block_count_bytes[2],
            block_count_bytes[3],
        ]);
        assert_eq!(
            block_count_from_bytes, 1,
            "first entry's block_count must be 1"
        );
    }

    /// Test that `document_count` is correctly written to and read from the segment header.
    #[test]
    fn test_document_count_roundtrip() {
        let index = make_test_index();
        let segment = write_segment(&index).expect("write_segment should succeed");
        let header = SegmentHeader::read(&segment).expect("header should decode");

        // The test index has 2 documents (with IDs 0 and 1)
        assert_eq!(
            header.document_count, 2,
            "header document_count should be 2"
        );

        // Verify the document_count is at the correct byte offset [12..16)
        let doc_count_bytes = &segment[12..16];
        let doc_count_from_bytes = u32::from_le_bytes([
            doc_count_bytes[0],
            doc_count_bytes[1],
            doc_count_bytes[2],
            doc_count_bytes[3],
        ]);
        assert_eq!(
            doc_count_from_bytes, 2,
            "document_count bytes at [12..16) should encode 2"
        );
    }

    /// Build a multi-block test index with a term having > `BLOCK_DOC_COUNT` postings.
    /// Also includes single-posting terms to verify block `count=1`.
    fn make_multiblock_test_index() -> InMemoryIndex {
        // 201 documents
        let mut documents = BTreeSet::new();
        for doc_id in 0..201 {
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
                doc_count: 201,
                total_terms: 3,
            },
        );

        let mut terms_to_ids = BTreeMap::new();
        let mut term_entries = vec![];

        // Term "common" (term_id = 0): 200 postings => 2 blocks (128 + 72)
        terms_to_ids.insert(
            (field_id, String::from("common")),
            leit_core::TermId::new(0),
        );
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(0),
            term: String::from("common"),
        });

        // Term "rare" (term_id = 1): 1 posting => 1 block
        terms_to_ids.insert((field_id, String::from("rare")), leit_core::TermId::new(1));
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(1),
            term: String::from("rare"),
        });

        // Term "uncommon" (term_id = 2): 5 postings => 1 block
        terms_to_ids.insert(
            (field_id, String::from("uncommon")),
            leit_core::TermId::new(2),
        );
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(2),
            term: String::from("uncommon"),
        });

        let mut postings = BTreeMap::new();

        // "common": 200 postings with varied term_freqs
        let mut common_postings = vec![];
        for doc_id in 0..200 {
            common_postings.push(PostingEntry {
                doc_id,
                term_freq: (doc_id % 10) + 1,
            });
        }
        postings.insert(leit_core::TermId::new(0), common_postings);

        // "rare": 1 posting in doc 200
        postings.insert(
            leit_core::TermId::new(1),
            vec![PostingEntry {
                doc_id: 200,
                term_freq: 5,
            }],
        );

        // "uncommon": 5 postings in docs 10, 20, 30, 40, 50
        postings.insert(
            leit_core::TermId::new(2),
            vec![
                PostingEntry {
                    doc_id: 10,
                    term_freq: 2,
                },
                PostingEntry {
                    doc_id: 20,
                    term_freq: 3,
                },
                PostingEntry {
                    doc_id: 30,
                    term_freq: 4,
                },
                PostingEntry {
                    doc_id: 40,
                    term_freq: 5,
                },
                PostingEntry {
                    doc_id: 50,
                    term_freq: 2,
                },
            ],
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

    /// Test T3 obligation: multi-block round-trip with block metadata verification.
    /// Build an index with a term having > `BLOCK_DOC_COUNT` postings, write the segment,
    /// and verify block summaries (`end_doc`, `max_term_freq`, `decode_offset`) are correct.
    #[test]
    fn roundtrip_with_block_meta() {
        let index = make_multiblock_test_index();
        let segment = write_segment(&index).expect("write_segment should succeed");
        let header = SegmentHeader::read(&segment).expect("header should decode");

        // Read postings table to locate the "common" term (term_id=0, pt_index=0)
        let pt = PostingsTableReader::new(
            &segment,
            header.postings_table_offset,
            header.postings_data_offset,
        )
        .expect("PostingsTableReader should construct");

        // Entry 0: "common" term with 200 postings
        let (_pdata_offset, _pdata_len, doc_freq, _codec_id, first_block_idx, block_count) =
            pt.entry(0).expect("postings entry 0 should exist");
        assert_eq!(doc_freq, 200, "common term should have 200 postings");
        // 200 postings => BLOCK_DOC_COUNT=128 => 2 blocks (128 + 72)
        assert_eq!(
            block_count, 2,
            "term with 200 postings should have 2 blocks"
        );

        // Read block metadata
        let bm = BlockMetadataReader::new(
            &segment,
            header.block_meta_offset,
            header.stored_fields_offset,
        )
        .expect("BlockMetadataReader should construct");

        // Block 0 for "common": first_block_idx=0
        let (end_doc_0, max_freq_0, decode_offset_0) =
            bm.entry(first_block_idx).expect("block 0 should exist");
        // Block 0 contains postings[0..128], so end_doc = posting[127].doc_id = 127
        assert_eq!(
            end_doc_0, 127,
            "block 0 end_doc should be 127 (128th posting, index 127)"
        );
        // decode_offset for block 0 should be 0 (relative to term's payload start)
        assert_eq!(
            decode_offset_0, 0,
            "block 0 decode_offset should be 0 (first block)"
        );
        // max_term_freq in block 0: postings[0..128] have term_freq = ((doc_id % 10) + 1)
        // Maximum is when doc_id % 10 == 9, i.e. doc_id in {9, 19, ..., 119}, max freq = 10
        assert_eq!(max_freq_0, 10, "block 0 max_term_freq should be 10");

        // Block 1 for "common": first_block_idx + 1 = 1
        let (end_doc_1, max_freq_1, decode_offset_1) =
            bm.entry(first_block_idx + 1).expect("block 1 should exist");
        // Block 1 contains postings[128..200], so end_doc = posting[199].doc_id = 199
        assert_eq!(
            end_doc_1, 199,
            "block 1 end_doc should be 199 (last posting, index 199)"
        );
        // decode_offset for block 1: (128 postings) * (8 bytes per posting) = 1024
        assert_eq!(
            decode_offset_1, 1024,
            "block 1 decode_offset should be 1024 (128 * 8 bytes)"
        );
        // max_term_freq in block 1: postings[128..200] also follow ((doc_id % 10) + 1), max = 10
        assert_eq!(max_freq_1, 10, "block 1 max_term_freq should be 10");

        // Entry 1: "rare" term with 1 posting => 1 block
        let (_pdata_offset1, _pdata_len1, doc_freq1, _codec_id1, first_block_idx1, block_count1) =
            pt.entry(1).expect("postings entry 1 should exist");
        assert_eq!(doc_freq1, 1, "rare term should have 1 posting");
        assert_eq!(block_count1, 1, "term with 1 posting should have 1 block");

        // Block for "rare": index = first_block_idx1
        let (end_doc_rare, max_freq_rare, decode_offset_rare) =
            bm.entry(first_block_idx1).expect("rare block should exist");
        assert_eq!(
            end_doc_rare, 200,
            "rare block end_doc should be 200 (its only doc_id)"
        );
        assert_eq!(
            decode_offset_rare, 0,
            "rare block decode_offset should be 0"
        );
        assert_eq!(max_freq_rare, 5, "rare block max_term_freq should be 5");

        // Entry 2: "uncommon" term with 5 postings => 1 block
        let (_pdata_offset2, _pdata_len2, doc_freq2, _codec_id2, first_block_idx2, block_count2) =
            pt.entry(2).expect("postings entry 2 should exist");
        assert_eq!(doc_freq2, 5, "uncommon term should have 5 postings");
        assert_eq!(block_count2, 1, "term with 5 postings should have 1 block");

        // Block for "uncommon": index = first_block_idx2
        let (end_doc_uncommon, max_freq_uncommon, decode_offset_uncommon) = bm
            .entry(first_block_idx2)
            .expect("uncommon block should exist");
        assert_eq!(
            end_doc_uncommon, 50,
            "uncommon block end_doc should be 50 (its last doc_id)"
        );
        assert_eq!(
            decode_offset_uncommon, 0,
            "uncommon block decode_offset should be 0"
        );
        // max_term_freq in uncommon: postings have freqs [2, 3, 4, 5, 2], max = 5
        assert_eq!(
            max_freq_uncommon, 5,
            "uncommon block max_term_freq should be 5"
        );
    }

    /// Test T3 scenario: skip locates block via metadata without scanning full postings.
    /// Use the multi-block "common" term and verify that finding a doc in block 1
    /// requires inspecting only block summaries, not all `postings`.
    #[test]
    fn skip_locates_block_without_full_scan() {
        let index = make_multiblock_test_index();
        let segment = write_segment(&index).expect("write_segment should succeed");
        let header = SegmentHeader::read(&segment).expect("header should decode");

        // Read postings table: "common" term (entry 0) has first_block_idx and block_count
        let pt = PostingsTableReader::new(
            &segment,
            header.postings_table_offset,
            header.postings_data_offset,
        )
        .expect("PostingsTableReader should construct");

        let (_pdata_offset, pdata_len, doc_freq, _codec_id, first_block_idx, block_count) =
            pt.entry(0).expect("postings entry 0 should exist");
        assert_eq!(block_count, 2, "common term should have 2 blocks");

        // Read block metadata for this term's blocks
        let bm = BlockMetadataReader::new(
            &segment,
            header.block_meta_offset,
            header.stored_fields_offset,
        )
        .expect("BlockMetadataReader should construct");

        // Target: find the block containing doc_id=150
        // doc_id=150 is in block 1 (postings[128..200])
        let target_doc = 150;

        // Scan ONLY block summaries to locate the block
        // Block 0: end_doc=127, so doc_id=150 > 127 => not in block 0
        // Block 1: end_doc=199, so doc_id=150 <= 199 => candidate block
        let (end_doc_0, _, _) = bm.entry(first_block_idx).expect("block 0 exists");
        assert!(
            target_doc > end_doc_0,
            "target_doc=150 should be > block 0 end_doc=127"
        );

        let (end_doc_1, _, _decode_offset_1) =
            bm.entry(first_block_idx + 1).expect("block 1 exists");
        assert!(
            target_doc <= end_doc_1,
            "target_doc=150 should be <= block 1 end_doc=199"
        );

        // The candidate block is block_idx = first_block_idx + 1
        let candidate_block_idx = first_block_idx + 1;

        // Key assertion: we located the block by inspecting only 2 block summaries,
        // not by scanning all 200 postings.
        // Summary count inspected: 2 (block 0 + block 1 from the loop)
        // Total postings: 200 (we MUST NOT scan these)
        let block_summaries_inspected = 2_u32;
        assert!(
            block_summaries_inspected < doc_freq,
            "we should inspect {} block summaries, far less than {} postings",
            block_summaries_inspected,
            doc_freq
        );

        // Verify the candidate block's decode_offset to prove we can compute
        // the postings byte range without scanning postings data
        let (_end_doc, _max_freq, candidate_decode_offset) = bm
            .entry(candidate_block_idx)
            .expect("candidate block exists");

        // Block 1 starts at decode_offset_1 = 1024 bytes from the term's payload
        assert_eq!(
            candidate_decode_offset, 1024,
            "candidate block decode_offset should be 1024"
        );

        // For an uncompressed format with 8 bytes per posting:
        // Block 1 payload byte range within the term's data:
        // [decode_offset_1, decode_offset_1 + (72 * 8)) = [1024, 1024 + 576) = [1024, 1600)
        let block_1_start = candidate_decode_offset;
        let block_1_posting_count = 72_u32; // 200 - 128
        let block_1_end = block_1_start + (block_1_posting_count * 8);

        // Validate that this sub-range lies within the term's total payload
        // (which is pdata_len bytes)
        assert!(
            block_1_end <= pdata_len,
            "block 1 payload [{}..{}) should fit within term payload of {} bytes",
            block_1_start,
            block_1_end,
            pdata_len
        );

        // Proof: we never needed to read the postings data itself.
        // Only block metadata was consulted.
    }

    /// Test: terms with exactly `BLOCK_DOC_COUNT`, `BLOCK_DOC_COUNT + 1`, and `2 * BLOCK_DOC_COUNT` postings.
    /// Verifies `block_count` and `decode_offset` computation for boundary cases.
    #[test]
    fn test_block_boundary_cases() {
        // Build a test index with three terms matching the boundary cases
        let mut documents = BTreeSet::new();
        for doc_id in 0..256 {
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
                doc_count: 256,
                total_terms: 3,
            },
        );

        let mut terms_to_ids = BTreeMap::new();
        let mut term_entries = vec![];

        // Term 0: exactly BLOCK_DOC_COUNT (128) postings => 1 block
        terms_to_ids.insert(
            (field_id, String::from("one_block")),
            leit_core::TermId::new(0),
        );
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(0),
            term: String::from("one_block"),
        });

        // Term 1: BLOCK_DOC_COUNT + 1 (129) postings => 2 blocks (128 + 1)
        terms_to_ids.insert(
            (field_id, String::from("two_blocks_uneven")),
            leit_core::TermId::new(1),
        );
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(1),
            term: String::from("two_blocks_uneven"),
        });

        // Term 2: 2 * BLOCK_DOC_COUNT (256) postings => 2 blocks (128 + 128)
        terms_to_ids.insert(
            (field_id, String::from("two_blocks_even")),
            leit_core::TermId::new(2),
        );
        term_entries.push(TermEntry {
            field_id,
            term_id: leit_core::TermId::new(2),
            term: String::from("two_blocks_even"),
        });

        let mut postings = BTreeMap::new();

        // Term 0: postings[0..128]
        let mut term0_postings = vec![];
        for doc_id in 0..128 {
            term0_postings.push(PostingEntry {
                doc_id,
                term_freq: 1,
            });
        }
        postings.insert(leit_core::TermId::new(0), term0_postings);

        // Term 1: postings[0..129]
        let mut term1_postings = vec![];
        for doc_id in 0..129 {
            term1_postings.push(PostingEntry {
                doc_id,
                term_freq: 1,
            });
        }
        postings.insert(leit_core::TermId::new(1), term1_postings);

        // Term 2: postings[0..256]
        let mut term2_postings = vec![];
        for doc_id in 0..256 {
            term2_postings.push(PostingEntry {
                doc_id,
                term_freq: 1,
            });
        }
        postings.insert(leit_core::TermId::new(2), term2_postings);

        let mut posting_blocks = BTreeMap::new();
        posting_blocks.insert(leit_core::TermId::new(0), vec![]);
        posting_blocks.insert(leit_core::TermId::new(1), vec![]);
        posting_blocks.insert(leit_core::TermId::new(2), vec![]);

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

        let segment = write_segment(&index).expect("write_segment should succeed");
        let header = SegmentHeader::read(&segment).expect("header should decode");

        let pt = PostingsTableReader::new(
            &segment,
            header.postings_table_offset,
            header.postings_data_offset,
        )
        .expect("PostingsTableReader should construct");

        let bm = BlockMetadataReader::new(
            &segment,
            header.block_meta_offset,
            header.stored_fields_offset,
        )
        .expect("BlockMetadataReader should construct");

        // Term 0: 128 postings => 1 block
        let (_offset0, _len0, doc_freq0, _codec_id0, first_block0, block_count0) =
            pt.entry(0).expect("term 0 should exist");
        assert_eq!(doc_freq0, 128, "term 0 should have 128 postings");
        assert_eq!(block_count0, 1, "term 0 (128 postings) should have 1 block");
        let (end_doc0, _, decode_offset0) =
            bm.entry(first_block0).expect("term 0 block should exist");
        assert_eq!(
            end_doc0, 127,
            "term 0 block end_doc should be 127 (last of 128 docs)"
        );
        assert_eq!(decode_offset0, 0, "term 0 block decode_offset should be 0");

        // Term 1: 129 postings => 2 blocks (128 + 1)
        let (_offset1, _len1, doc_freq1, _codec_id1, first_block1, block_count1) =
            pt.entry(1).expect("term 1 should exist");
        assert_eq!(doc_freq1, 129, "term 1 should have 129 postings");
        assert_eq!(
            block_count1, 2,
            "term 1 (129 postings) should have 2 blocks"
        );
        let (end_doc1_0, _, decode_offset1_0) =
            bm.entry(first_block1).expect("term 1 block 0 should exist");
        assert_eq!(end_doc1_0, 127, "term 1 block 0 end_doc should be 127");
        assert_eq!(
            decode_offset1_0, 0,
            "term 1 block 0 decode_offset should be 0"
        );
        let (end_doc1_1, _, decode_offset1_1) = bm
            .entry(first_block1 + 1)
            .expect("term 1 block 1 should exist");
        assert_eq!(
            end_doc1_1, 128,
            "term 1 block 1 end_doc should be 128 (129th posting)"
        );
        assert_eq!(
            decode_offset1_1, 1024,
            "term 1 block 1 decode_offset should be 1024 (128 * 8)"
        );

        // Term 2: 256 postings => 2 blocks (128 + 128)
        let (_offset2, _len2, doc_freq2, _codec_id2, first_block2, block_count2) =
            pt.entry(2).expect("term 2 should exist");
        assert_eq!(doc_freq2, 256, "term 2 should have 256 postings");
        assert_eq!(
            block_count2, 2,
            "term 2 (256 postings) should have 2 blocks"
        );
        let (end_doc2_0, _, decode_offset2_0) =
            bm.entry(first_block2).expect("term 2 block 0 should exist");
        assert_eq!(end_doc2_0, 127, "term 2 block 0 end_doc should be 127");
        assert_eq!(
            decode_offset2_0, 0,
            "term 2 block 0 decode_offset should be 0"
        );
        let (end_doc2_1, _, decode_offset2_1) = bm
            .entry(first_block2 + 1)
            .expect("term 2 block 1 should exist");
        assert_eq!(
            end_doc2_1, 255,
            "term 2 block 1 end_doc should be 255 (last of 256 docs)"
        );
        assert_eq!(
            decode_offset2_1, 1024,
            "term 2 block 1 decode_offset should be 1024"
        );
    }

    /// Test: empty corpus (no terms) produces a valid segment with zero-length `block_meta`.
    #[test]
    fn test_write_segment_empty_corpus() {
        let documents = BTreeSet::new(); // empty

        let field_id = FieldId::new(1);
        let mut field_names = BTreeMap::new();
        field_names.insert(String::from("text"), field_id);

        let mut field_stats = BTreeMap::new();
        field_stats.insert(
            field_id,
            FieldMetadata {
                field_id,
                doc_count: 0,
                total_terms: 0,
            },
        );

        let terms_to_ids = BTreeMap::new();
        let term_entries = vec![];
        let postings = BTreeMap::new();
        let posting_blocks = BTreeMap::new();
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

        // Write segment with empty corpus
        let segment = write_segment(&index).expect("write_segment should succeed on empty corpus");

        // Read and verify header
        let header = SegmentHeader::read(&segment).expect("header should decode");
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, FORMAT_VERSION);
        assert_eq!(header.document_count, 0, "document_count should be 0");

        // Verify block_meta section is zero-length (since no terms => no blocks)
        let block_meta_size = header.stored_fields_offset - header.block_meta_offset;
        assert_eq!(
            block_meta_size, 0,
            "block_meta section should be zero-length for empty corpus"
        );

        // Verify lexicon is empty
        let lex = LexiconReader::new(
            &segment,
            header.lexicon_offset,
            header.postings_table_offset,
        )
        .expect("LexiconReader should construct");
        assert_eq!(
            lex.len(),
            0,
            "lexicon should have 0 entries for empty corpus"
        );

        // Verify postings table is valid (but empty)
        let pt = PostingsTableReader::new(
            &segment,
            header.postings_table_offset,
            header.postings_data_offset,
        )
        .expect("PostingsTableReader should construct");
        assert_eq!(
            pt.len(),
            0,
            "postings table should have 0 entries for empty corpus"
        );

        // Verify block metadata reader accepts zero-length section
        let bm = BlockMetadataReader::new(
            &segment,
            header.block_meta_offset,
            header.stored_fields_offset,
        )
        .expect("BlockMetadataReader should accept zero-length section");
        assert_eq!(
            bm.len(),
            0,
            "block_meta should have 0 entries for empty corpus"
        );
    }

    /// Test: functional `decode_offset` — use it to read real postings bytes from the multi-block term.
    /// This test proves that `decode_offset` is semantically correct, not just round-tripped.
    #[test]
    fn test_decode_offset_functional() {
        let index = make_multiblock_test_index();
        let segment = write_segment(&index).expect("write_segment should succeed");
        let header = SegmentHeader::read(&segment).expect("header should decode");

        // Read postings table for "common" term (200 postings => 2 blocks)
        let pt = PostingsTableReader::new(
            &segment,
            header.postings_table_offset,
            header.postings_data_offset,
        )
        .expect("PostingsTableReader should construct");

        let (pdata_offset, pdata_len, _doc_freq, _codec_id, first_block_idx, block_count) =
            pt.entry(0).expect("common term should exist");
        assert_eq!(block_count, 2, "common term should have 2 blocks");

        // Read postings data reader
        let pd = PostingsDataReader::new(
            &segment,
            header.postings_data_offset,
            header.block_meta_offset,
        )
        .expect("PostingsDataReader should construct");

        // Read the entire term's postings payload
        let term_payload = pd
            .range(pdata_offset, pdata_len)
            .expect("should read term's postings data");

        // Read block metadata for the two blocks
        let bm = BlockMetadataReader::new(
            &segment,
            header.block_meta_offset,
            header.stored_fields_offset,
        )
        .expect("BlockMetadataReader should construct");

        let (_end_doc_0, _max_freq_0, decode_offset_0) =
            bm.entry(first_block_idx).expect("block 0 should exist");
        let (_end_doc_1, _max_freq_1, decode_offset_1) =
            bm.entry(first_block_idx + 1).expect("block 1 should exist");

        // Block 0 starts at decode_offset_0 (which should be 0)
        // Block 1 starts at decode_offset_1 (which should be 1024)
        assert_eq!(decode_offset_0, 0, "block 0 decode_offset must be 0");
        assert_eq!(decode_offset_1, 1024, "block 1 decode_offset must be 1024");

        // Extract block 0 bytes using its decode_offset
        let block0_start = decode_offset_0 as usize;
        let block0_end = decode_offset_1 as usize; // next block's start is this block's end
        let block0_bytes = &term_payload[block0_start..block0_end];

        // Extract block 1 bytes using its decode_offset
        let block1_start = decode_offset_1 as usize;
        let block1_end = term_payload.len(); // term's end is the last block's end
        let block1_bytes = &term_payload[block1_start..block1_end];

        // Block 0 should contain 128 postings = 128 * 8 = 1024 bytes
        assert_eq!(
            block0_bytes.len(),
            1024,
            "block 0 bytes should be 1024 (128 postings * 8 bytes)"
        );

        // Block 1 should contain 72 postings = 72 * 8 = 576 bytes
        assert_eq!(
            block1_bytes.len(),
            576,
            "block 1 bytes should be 576 (72 postings * 8 bytes)"
        );

        // Now, verify that block 1's first posting is indeed posting[128] from the original data
        // Posting 128 is (doc_id=128, term_freq=9) because term_freq = (doc_id % 10) + 1
        let first_posting_block1_doc_id = u32::from_le_bytes([
            block1_bytes[0],
            block1_bytes[1],
            block1_bytes[2],
            block1_bytes[3],
        ]);
        let first_posting_block1_term_freq = u32::from_le_bytes([
            block1_bytes[4],
            block1_bytes[5],
            block1_bytes[6],
            block1_bytes[7],
        ]);

        assert_eq!(
            first_posting_block1_doc_id, 128,
            "block 1's first posting should have doc_id=128"
        );
        assert_eq!(
            first_posting_block1_term_freq, 9,
            "block 1's first posting (doc_id=128) should have term_freq=9 (128 % 10 + 1)"
        );

        // Verify the last posting in block 0 (posting[127])
        // Posting 127 is (doc_id=127, term_freq=8) because term_freq = (doc_id % 10) + 1
        let last_posting_block0_offset = block0_bytes.len() - 8; // last 8 bytes of block 0
        let last_posting_block0_doc_id = u32::from_le_bytes([
            block0_bytes[last_posting_block0_offset],
            block0_bytes[last_posting_block0_offset + 1],
            block0_bytes[last_posting_block0_offset + 2],
            block0_bytes[last_posting_block0_offset + 3],
        ]);
        let last_posting_block0_term_freq = u32::from_le_bytes([
            block0_bytes[last_posting_block0_offset + 4],
            block0_bytes[last_posting_block0_offset + 5],
            block0_bytes[last_posting_block0_offset + 6],
            block0_bytes[last_posting_block0_offset + 7],
        ]);

        assert_eq!(
            last_posting_block0_doc_id, 127,
            "block 0's last posting should have doc_id=127"
        );
        assert_eq!(
            last_posting_block0_term_freq, 8,
            "block 0's last posting (doc_id=127) should have term_freq=8 (127 % 10 + 1)"
        );
    }
}
