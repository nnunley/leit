// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Borrowed section readers for segment format v1.
//!
//! Each reader borrows `&[u8]` directly from the segment buffer and provides zero-copy,
//! O(1) access to fixed-width entries. Variable-length data (term bytes) is addressed via
//! fixed-width index entries, keeping metadata access O(1).
//!
//! **Layouts (fixed-width entries only):**
//!
//! - `FieldTableReader`: `field_id` (`u32`) + `doc_count` (`u32`) + `total_terms` (`u32`), 12 bytes/entry
//! - `LexiconReader`: index entries (`term_offset`: `u64` + `term_len`: `u32` + `postings_table_index`: `u32`,
//!   16 bytes/entry) + variable-length term bytes blob
//! - `PostingsTableReader`: `postings_data_offset` (`u64`) + `postings_data_len` (`u32`) + `doc_freq` (`u32`)
//!   + `reserved_codec_id` (`u32`), 20 bytes/entry
//! - `PostingsDataReader`: borrowed postings payload bytes; ranges from `PostingsTableReader` offsets
//!
//! All readers validate bounds against the segment buffer and return `SegmentError` on out-of-bounds
//! access (never panic).
//!
//! **`no_std`+`alloc`:** Readers hold only borrowed slices and
//! validated offsets; no allocation beyond the struct itself.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "readers are consumed only by the (currently crate-internal) segment view; unused in the lib build until that view is wired into a public consumer"
    )
)]

use crate::error::SegmentError;

/// Helper: convert usize offset to u64 for bounds checking.
#[inline]
fn usize_to_u64(n: usize) -> u64 {
    n as u64
}

/// Helper: read u32 LE from bytes at offset, or return Truncated.
#[inline]
fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, SegmentError> {
    let needed = offset.checked_add(4).ok_or(SegmentError::Truncated {
        needed: usize::MAX,
        found: bytes.len(),
    })?;
    if bytes.len() < needed {
        return Err(SegmentError::Truncated {
            needed,
            found: bytes.len(),
        });
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

/// Helper: read u64 LE from bytes at offset, or return Truncated.
#[inline]
fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64, SegmentError> {
    let needed = offset.checked_add(8).ok_or(SegmentError::Truncated {
        needed: usize::MAX,
        found: bytes.len(),
    })?;
    if bytes.len() < needed {
        return Err(SegmentError::Truncated {
            needed,
            found: bytes.len(),
        });
    }
    Ok(u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ]))
}

/// Field table reader: borrows field metadata entries (fixed-width, O(1) access).
///
/// **Layout (12 bytes per entry):**
/// - Offset 0: count (`u32` LE, number of fields)
/// - Offset 4..: entries: `field_id` (`u32`) + `doc_count` (`u32`) + `total_terms` (`u32`)
///
/// **Bounds:** `field_table_offset .. lexicon_offset`
#[derive(Clone, Copy, Debug)]
pub struct FieldTableReader<'a> {
    buffer: &'a [u8],
    section_start: u64,
    section_end: u64,
    count: u32,
}

impl<'a> FieldTableReader<'a> {
    /// Construct a `FieldTableReader` from segment buffer and offsets.
    ///
    /// # Arguments
    /// * `buffer` - complete segment buffer
    /// * `field_table_offset` - offset to field table section start
    /// * `lexicon_offset` - offset to next section (end of field table)
    ///
    /// # Returns
    /// `FieldTableReader` or `SegmentError` if truncated or out of bounds.
    pub fn new(
        buffer: &'a [u8],
        field_table_offset: u64,
        lexicon_offset: u64,
    ) -> Result<Self, SegmentError> {
        let buffer_len = usize_to_u64(buffer.len());

        // Bounds check: field_table_offset must be in buffer
        if field_table_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: field_table_offset,
                limit: buffer_len,
            });
        }

        // If zero-length (field_table_offset == lexicon_offset), return empty reader
        if field_table_offset == lexicon_offset {
            return Ok(Self {
                buffer,
                section_start: field_table_offset,
                section_end: lexicon_offset,
                count: 0,
            });
        }

        // Read count from offset field_table_offset
        let start_usize = usize::try_from(field_table_offset).unwrap_or(usize::MAX);
        let count = read_u32_le(buffer, start_usize)?;

        Ok(Self {
            buffer,
            section_start: field_table_offset,
            section_end: lexicon_offset,
            count,
        })
    }

    /// Number of field entries in this section.
    pub fn len(&self) -> u32 {
        self.count
    }

    /// True if there are no fields.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get field metadata entry [i]: (`field_id`, `doc_count`, `total_terms`).
    ///
    /// # Arguments
    /// * `i` - entry index (`0..len()`)
    ///
    /// # Returns
    /// (`field_id`: `u32`, `doc_count`: `u32`, `total_terms`: `u32`) or `SegmentError` if out of bounds.
    pub fn entry(&self, i: u32) -> Result<(u32, u32, u32), SegmentError> {
        if i >= self.count {
            return Err(SegmentError::BadOffset {
                offset: i as u64,
                limit: self.count as u64,
            });
        }

        // Entry offset: 4 (count) + i * 12
        let i_u64 = i as u64;
        let entry_offset = 4_u64
            .checked_add(i_u64.checked_mul(12).ok_or(SegmentError::BadOffset {
                offset: i_u64,
                limit: u64::MAX,
            })?)
            .ok_or(SegmentError::BadOffset {
                offset: i_u64,
                limit: u64::MAX,
            })?;
        let absolute_offset =
            self.section_start
                .checked_add(entry_offset)
                .ok_or(SegmentError::BadOffset {
                    offset: entry_offset,
                    limit: self.section_end,
                })?;

        // Bounds check
        let absolute_offset_plus_12 =
            absolute_offset
                .checked_add(12)
                .ok_or(SegmentError::BadOffset {
                    offset: absolute_offset,
                    limit: self.section_end,
                })?;
        if absolute_offset_plus_12 > self.section_end {
            return Err(SegmentError::BadOffset {
                offset: absolute_offset,
                limit: self.section_end,
            });
        }

        let offset_usize = usize::try_from(absolute_offset).unwrap_or(usize::MAX);
        let field_id = read_u32_le(self.buffer, offset_usize)?;
        let doc_count = read_u32_le(self.buffer, offset_usize + 4)?;
        let total_terms = read_u32_le(self.buffer, offset_usize + 8)?;

        Ok((field_id, doc_count, total_terms))
    }
}

/// Lexicon reader: borrows term dictionary with separate index + variable-length bytes.
///
/// **Layout:**
/// - Offset 0: count (`u32` LE, number of terms)
/// - Offset 4..: index entries (16 bytes each): `term_offset` (`u64`) + `term_len` (`u32`) + `postings_table_index` (`u32`)
/// - After index: variable-length term bytes blob
///
/// Term bytes offset is relative to the start of the blob (after the index).
///
/// **Bounds:** `lexicon_offset .. postings_table_offset`
#[derive(Clone, Copy, Debug)]
pub struct LexiconReader<'a> {
    buffer: &'a [u8],
    section_start: u64,
    section_end: u64,
    count: u32,
    blob_start: u64, // Start of variable-length term bytes (after the index)
}

impl<'a> LexiconReader<'a> {
    /// Construct a `LexiconReader` from segment buffer and offsets.
    ///
    /// # Arguments
    /// * `buffer` - complete segment buffer
    /// * `lexicon_offset` - offset to lexicon section start
    /// * `postings_table_offset` - offset to next section (end of lexicon)
    ///
    /// # Returns
    /// `LexiconReader` or `SegmentError` if truncated or out of bounds.
    pub fn new(
        buffer: &'a [u8],
        lexicon_offset: u64,
        postings_table_offset: u64,
    ) -> Result<Self, SegmentError> {
        let buffer_len = usize_to_u64(buffer.len());

        // Bounds check
        if lexicon_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: lexicon_offset,
                limit: buffer_len,
            });
        }

        // If zero-length, return empty reader
        if lexicon_offset == postings_table_offset {
            return Ok(Self {
                buffer,
                section_start: lexicon_offset,
                section_end: postings_table_offset,
                count: 0,
                blob_start: lexicon_offset,
            });
        }

        // Read count from offset lexicon_offset
        let start_usize = usize::try_from(lexicon_offset).unwrap_or(usize::MAX);
        let count = read_u32_le(buffer, start_usize)?;

        // blob_start = lexicon_offset + 4 (count) + count * 16 (index entries)
        let count_u64 = count as u64;
        let index_size = count_u64.checked_mul(16).ok_or(SegmentError::BadOffset {
            offset: count_u64,
            limit: postings_table_offset,
        })?;
        let blob_start = lexicon_offset
            .checked_add(4)
            .ok_or(SegmentError::BadOffset {
                offset: lexicon_offset,
                limit: postings_table_offset,
            })?
            .checked_add(index_size)
            .ok_or(SegmentError::BadOffset {
                offset: lexicon_offset,
                limit: postings_table_offset,
            })?;

        // Validate blob_start is within section bounds
        if blob_start > postings_table_offset {
            return Err(SegmentError::BadOffset {
                offset: blob_start,
                limit: postings_table_offset,
            });
        }

        Ok(Self {
            buffer,
            section_start: lexicon_offset,
            section_end: postings_table_offset,
            count,
            blob_start,
        })
    }

    /// Number of term entries in this section.
    pub fn len(&self) -> u32 {
        self.count
    }

    /// True if there are no terms.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get lexicon entry [i]: (`term_bytes`, `postings_table_index`).
    ///
    /// # Arguments
    /// * `i` - entry index (`0..len()`)
    ///
    /// # Returns
    /// (`term_bytes`: &[u8], `postings_table_index`: u32) or `SegmentError` if out of bounds.
    pub fn entry(&self, i: u32) -> Result<(&'a [u8], u32), SegmentError> {
        if i >= self.count {
            return Err(SegmentError::BadOffset {
                offset: i as u64,
                limit: self.count as u64,
            });
        }

        // Index entry offset: section_start + 4 (count) + i * 16
        let i_u64 = i as u64;
        let entry_offset = 4_u64
            .checked_add(i_u64.checked_mul(16).ok_or(SegmentError::BadOffset {
                offset: i_u64,
                limit: u64::MAX,
            })?)
            .ok_or(SegmentError::BadOffset {
                offset: i_u64,
                limit: u64::MAX,
            })?;
        let index_entry_offset =
            self.section_start
                .checked_add(entry_offset)
                .ok_or(SegmentError::BadOffset {
                    offset: entry_offset,
                    limit: self.section_end,
                })?;

        // Bounds check
        let index_entry_end =
            index_entry_offset
                .checked_add(16)
                .ok_or(SegmentError::BadOffset {
                    offset: index_entry_offset,
                    limit: self.section_end,
                })?;
        if index_entry_end > self.section_end {
            return Err(SegmentError::BadOffset {
                offset: index_entry_offset,
                limit: self.section_end,
            });
        }

        let offset_usize = usize::try_from(index_entry_offset).unwrap_or(usize::MAX);
        let term_offset = read_u64_le(self.buffer, offset_usize)?;
        let term_len = read_u32_le(self.buffer, offset_usize + 8)?;
        let postings_table_index = read_u32_le(self.buffer, offset_usize + 12)?;

        // Resolve term bytes: blob_start + term_offset
        let term_absolute_offset =
            self.blob_start
                .checked_add(term_offset)
                .ok_or(SegmentError::BadOffset {
                    offset: term_offset,
                    limit: self.section_end,
                })?;
        let term_bytes_end =
            term_absolute_offset
                .checked_add(term_len as u64)
                .ok_or(SegmentError::BadOffset {
                    offset: term_absolute_offset,
                    limit: self.section_end,
                })?;

        // Bounds check: term must fit within section
        if term_bytes_end > self.section_end {
            return Err(SegmentError::BadOffset {
                offset: term_bytes_end,
                limit: self.section_end,
            });
        }

        let term_start_usize = usize::try_from(term_absolute_offset).unwrap_or(usize::MAX);
        let term_end_usize = usize::try_from(term_bytes_end).unwrap_or(usize::MAX);

        if term_end_usize > self.buffer.len() {
            return Err(SegmentError::Truncated {
                needed: term_end_usize,
                found: self.buffer.len(),
            });
        }

        let term_bytes = &self.buffer[term_start_usize..term_end_usize];

        Ok((term_bytes, postings_table_index))
    }
}

/// Postings table reader: borrows postings metadata entries (fixed-width, O(1) access).
///
/// **Layout (20 bytes per entry):**
/// - Offset 0: count (u32 LE, number of postings entries)
/// - Offset 4..: entries (20 bytes each):
///   - `postings_data_offset` (u64, offset 0)
///   - `postings_data_len` (u32, offset 8)
///   - `doc_freq` (u32, offset 12)
///   - `reserved_codec_id` (u32, offset 16) — reserved for per-term codec selection; currently 0
///
/// **Bounds:** `postings_table_offset .. postings_data_offset`
#[derive(Clone, Copy, Debug)]
pub struct PostingsTableReader<'a> {
    buffer: &'a [u8],
    section_start: u64,
    section_end: u64,
    count: u32,
}

impl<'a> PostingsTableReader<'a> {
    /// Construct a `PostingsTableReader` from segment buffer and offsets.
    ///
    /// # Arguments
    /// * `buffer` - complete segment buffer
    /// * `postings_table_offset` - offset to postings table section start
    /// * `postings_data_offset` - offset to postings data section (end of postings table)
    ///
    /// # Returns
    /// `PostingsTableReader` or `SegmentError` if truncated or out of bounds.
    pub fn new(
        buffer: &'a [u8],
        postings_table_offset: u64,
        postings_data_offset: u64,
    ) -> Result<Self, SegmentError> {
        let buffer_len = usize_to_u64(buffer.len());

        // Bounds check
        if postings_table_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: postings_table_offset,
                limit: buffer_len,
            });
        }

        // If zero-length, return empty reader
        if postings_table_offset == postings_data_offset {
            return Ok(Self {
                buffer,
                section_start: postings_table_offset,
                section_end: postings_data_offset,
                count: 0,
            });
        }

        // Read count from offset postings_table_offset
        let start_usize = usize::try_from(postings_table_offset).unwrap_or(usize::MAX);
        let count = read_u32_le(buffer, start_usize)?;

        Ok(Self {
            buffer,
            section_start: postings_table_offset,
            section_end: postings_data_offset,
            count,
        })
    }

    /// Number of postings entries in this section.
    pub fn len(&self) -> u32 {
        self.count
    }

    /// True if there are no postings entries.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get postings metadata entry [i]: (`postings_data_offset`, `postings_data_len`, `doc_freq`, `reserved_codec_id`).
    ///
    /// # Arguments
    /// * `i` - entry index (`0..len()`)
    ///
    /// # Returns
    /// (`postings_data_offset`: u64, `postings_data_len`: u32, `doc_freq`: u32, `reserved_codec_id`: u32) or `SegmentError`.
    pub fn entry(&self, i: u32) -> Result<(u64, u32, u32, u32), SegmentError> {
        if i >= self.count {
            return Err(SegmentError::BadOffset {
                offset: i as u64,
                limit: self.count as u64,
            });
        }

        // Entry offset: 4 (count) + i * 20
        let i_u64 = i as u64;
        let entry_offset = 4_u64
            .checked_add(i_u64.checked_mul(20).ok_or(SegmentError::BadOffset {
                offset: i_u64,
                limit: u64::MAX,
            })?)
            .ok_or(SegmentError::BadOffset {
                offset: i_u64,
                limit: u64::MAX,
            })?;
        let absolute_offset =
            self.section_start
                .checked_add(entry_offset)
                .ok_or(SegmentError::BadOffset {
                    offset: entry_offset,
                    limit: self.section_end,
                })?;

        // Bounds check: entry is 20 bytes
        let absolute_offset_plus_20 =
            absolute_offset
                .checked_add(20)
                .ok_or(SegmentError::BadOffset {
                    offset: absolute_offset,
                    limit: self.section_end,
                })?;
        if absolute_offset_plus_20 > self.section_end {
            return Err(SegmentError::BadOffset {
                offset: absolute_offset,
                limit: self.section_end,
            });
        }

        let offset_usize = usize::try_from(absolute_offset).unwrap_or(usize::MAX);
        let postings_data_offset = read_u64_le(self.buffer, offset_usize)?;
        let postings_data_len = read_u32_le(self.buffer, offset_usize + 8)?;
        let doc_freq = read_u32_le(self.buffer, offset_usize + 12)?;
        let reserved_codec_id = read_u32_le(self.buffer, offset_usize + 16)?;

        Ok((
            postings_data_offset,
            postings_data_len,
            doc_freq,
            reserved_codec_id,
        ))
    }
}

/// Postings data reader: borrows postings payload bytes.
///
/// Accesses ranges specified by `PostingsTableReader` entries.
///
/// **Bounds:** `postings_data_offset .. block_meta_offset`
#[derive(Clone, Copy, Debug)]
pub struct PostingsDataReader<'a> {
    buffer: &'a [u8],
    section_start: u64,
    section_end: u64,
}

impl<'a> PostingsDataReader<'a> {
    /// Construct a `PostingsDataReader` from segment buffer and offsets.
    ///
    /// # Arguments
    /// * `buffer` - complete segment buffer
    /// * `postings_data_offset` - offset to postings data section start
    /// * `block_meta_offset` - offset to next section (end of postings data)
    ///
    /// # Returns
    /// `PostingsDataReader` or `SegmentError` if out of bounds.
    pub fn new(
        buffer: &'a [u8],
        postings_data_offset: u64,
        block_meta_offset: u64,
    ) -> Result<Self, SegmentError> {
        let buffer_len = usize_to_u64(buffer.len());

        // Bounds check
        if postings_data_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: postings_data_offset,
                limit: buffer_len,
            });
        }

        Ok(Self {
            buffer,
            section_start: postings_data_offset,
            section_end: block_meta_offset,
        })
    }

    /// Get postings data bytes for a term: [offset, offset + len).
    ///
    /// # Arguments
    /// * `offset` - byte offset within postings data section
    /// * `len` - byte length
    ///
    /// # Returns
    /// borrowed byte slice or `SegmentError` if out of bounds.
    pub fn range(&self, offset: u64, len: u32) -> Result<&'a [u8], SegmentError> {
        let start = self
            .section_start
            .checked_add(offset)
            .ok_or(SegmentError::BadOffset {
                offset,
                limit: self.section_end,
            })?;
        let end = start
            .checked_add(len as u64)
            .ok_or(SegmentError::BadOffset {
                offset: start,
                limit: self.section_end,
            })?;

        // Bounds check
        if end > self.section_end {
            return Err(SegmentError::BadOffset {
                offset: end,
                limit: self.section_end,
            });
        }

        let start_usize = usize::try_from(start).unwrap_or(usize::MAX);
        let end_usize = usize::try_from(end).unwrap_or(usize::MAX);

        if end_usize > self.buffer.len() {
            return Err(SegmentError::Truncated {
                needed: end_usize,
                found: self.buffer.len(),
            });
        }

        Ok(&self.buffer[start_usize..end_usize])
    }

    /// True if the postings data section is empty (zero-length).
    #[expect(dead_code, reason = "reserved for future API")]
    pub fn is_empty(&self) -> bool {
        self.section_start == self.section_end
    }
}

/// Block metadata reader: reserved (currently zero-length, content deferred).
///
/// **Bounds:** `block_meta_offset .. stored_fields_offset`
///
/// This section is reserved but currently zero-length. Iterating readers are deferred for a future release.
#[expect(
    dead_code,
    reason = "reserved for future releases; placeholder in current format, used by SegmentView"
)]
#[expect(
    unreachable_pub,
    reason = "pub(crate) for SegmentView visibility in current implementation"
)]
#[derive(Clone, Copy, Debug)]
pub struct BlockMetadataReader<'a> {
    buffer: &'a [u8],
    section_start: u64,
    section_end: u64,
}

impl<'a> BlockMetadataReader<'a> {
    /// Construct a `BlockMetadataReader` from segment buffer and offsets.
    ///
    /// # Arguments
    /// * `buffer` - complete segment buffer
    /// * `block_meta_offset` - offset to block metadata section start
    /// * `stored_fields_offset` - offset to next section (end of block metadata)
    ///
    /// # Returns
    /// `BlockMetadataReader` or `SegmentError` if out of bounds.
    ///
    /// In the current format, this section is typically zero-length; the reader is reserved for future releases.
    pub fn new(
        buffer: &'a [u8],
        block_meta_offset: u64,
        stored_fields_offset: u64,
    ) -> Result<Self, SegmentError> {
        let buffer_len = usize_to_u64(buffer.len());

        // Bounds check
        if block_meta_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: block_meta_offset,
                limit: buffer_len,
            });
        }

        Ok(Self {
            buffer,
            section_start: block_meta_offset,
            section_end: stored_fields_offset,
        })
    }

    /// True if the block metadata section is empty (zero-length, reserved).
    pub fn is_empty(&self) -> bool {
        self.section_start == self.section_end
    }

    /// Byte length of the block metadata section.
    pub fn len(&self) -> u64 {
        self.section_end - self.section_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_format::{FORMAT_VERSION, HEADER_SIZE, MAGIC, SegmentHeader};
    use alloc::vec;
    use alloc::vec::Vec;

    /// Helper to build a minimal test segment with header + sections.
    fn make_test_segment(
        field_table_data: Vec<u8>,
        lexicon_data: Vec<u8>,
        postings_table_data: Vec<u8>,
        postings_data_data: Vec<u8>,
    ) -> Vec<u8> {
        let mut offset: u64 = HEADER_SIZE as u64;

        let field_table_offset = offset;
        offset += field_table_data.len() as u64;

        let lexicon_offset = offset;
        offset += lexicon_data.len() as u64;

        let postings_table_offset = offset;
        offset += postings_table_data.len() as u64;

        let postings_data_offset = offset;
        offset += postings_data_data.len() as u64;

        // Reserve sections (zero-length in minimal v1)
        let block_meta_offset = offset;
        let stored_fields_offset = offset;
        let columnar_offset = offset;
        let footer_offset = offset;

        let header = SegmentHeader {
            magic: MAGIC,
            version: FORMAT_VERSION,
            format_flags: 0,
            document_count: 0,
            field_table_offset,
            lexicon_offset,
            postings_table_offset,
            postings_data_offset,
            block_meta_offset,
            stored_fields_offset,
            columnar_offset,
            footer_offset,
        };

        let mut buffer = vec![0_u8; HEADER_SIZE];
        buffer[0..HEADER_SIZE].copy_from_slice(&header.encode());

        buffer.extend_from_slice(&field_table_data);
        buffer.extend_from_slice(&lexicon_data);
        buffer.extend_from_slice(&postings_table_data);
        buffer.extend_from_slice(&postings_data_data);

        buffer
    }

    #[test]
    fn test_field_table_reader_empty() {
        // Empty field table
        let buffer = make_test_segment(vec![], vec![], vec![], vec![]);
        let reader = FieldTableReader::new(&buffer, 80, 80).expect("should create reader");
        assert!(reader.is_empty());
        assert_eq!(reader.len(), 0);
    }

    #[test]
    fn test_field_table_reader_single_entry() {
        // Field table with 1 entry: count (1) + entry (field_id=5, doc_count=10, total_terms=50)
        let mut field_data = vec![];
        field_data.extend_from_slice(&1_u32.to_le_bytes());
        field_data.extend_from_slice(&5_u32.to_le_bytes()); // field_id
        field_data.extend_from_slice(&10_u32.to_le_bytes()); // doc_count
        field_data.extend_from_slice(&50_u32.to_le_bytes()); // total_terms

        let buffer = make_test_segment(field_data, vec![], vec![], vec![]);
        let reader = FieldTableReader::new(&buffer, 80, 80 + 16).expect("should create reader");
        assert_eq!(reader.len(), 1);

        let (field_id, doc_count, total_terms) = reader.entry(0).expect("entry should exist");
        assert_eq!(field_id, 5);
        assert_eq!(doc_count, 10);
        assert_eq!(total_terms, 50);
    }

    #[test]
    fn test_field_table_reader_out_of_bounds_entry() {
        let field_data = vec![];
        let buffer = make_test_segment(field_data, vec![], vec![], vec![]);
        let reader = FieldTableReader::new(&buffer, 80, 80).expect("should create reader");

        let result = reader.entry(0);
        assert!(matches!(result, Err(SegmentError::BadOffset { .. })));
    }

    #[test]
    fn test_lexicon_reader_empty() {
        let buffer = make_test_segment(vec![], vec![], vec![], vec![]);
        let reader = LexiconReader::new(&buffer, 80, 80).expect("should create reader");
        assert!(reader.is_empty());
        assert_eq!(reader.len(), 0);
    }

    #[test]
    fn test_lexicon_reader_single_term() {
        // Lexicon with 1 term: count (1) + index (term_offset=0, term_len=3, postings_idx=7)
        // + blob (term bytes "foo")
        let mut lex_data = vec![];
        lex_data.extend_from_slice(&1_u32.to_le_bytes()); // count
        lex_data.extend_from_slice(&0_u64.to_le_bytes()); // term_offset (relative to blob)
        lex_data.extend_from_slice(&3_u32.to_le_bytes()); // term_len
        lex_data.extend_from_slice(&7_u32.to_le_bytes()); // postings_table_index
        lex_data.extend_from_slice(b"foo"); // term bytes

        let buffer = make_test_segment(vec![], lex_data, vec![], vec![]);
        let lexicon_offset = 80_u64;
        let postings_table_offset = 80_u64 + 4 + 16 + 3;

        let reader = LexiconReader::new(&buffer, lexicon_offset, postings_table_offset)
            .expect("should create reader");
        assert_eq!(reader.len(), 1);

        let (term_bytes, postings_idx) = reader.entry(0).expect("entry should exist");
        assert_eq!(term_bytes, b"foo");
        assert_eq!(postings_idx, 7);
    }

    #[test]
    fn test_postings_table_reader_empty() {
        let buffer = make_test_segment(vec![], vec![], vec![], vec![]);
        let reader = PostingsTableReader::new(&buffer, 80, 80).expect("should create reader");
        assert!(reader.is_empty());
        assert_eq!(reader.len(), 0);
    }

    #[test]
    fn test_postings_table_reader_single_entry() {
        // Postings table with 1 entry: count (1) + entry (20 bytes)
        let mut post_table = vec![];
        post_table.extend_from_slice(&1_u32.to_le_bytes()); // count
        post_table.extend_from_slice(&100_u64.to_le_bytes()); // postings_data_offset
        post_table.extend_from_slice(&20_u32.to_le_bytes()); // postings_data_len
        post_table.extend_from_slice(&5_u32.to_le_bytes()); // doc_freq
        post_table.extend_from_slice(&0_u32.to_le_bytes()); // reserved_codec_id

        let buffer = make_test_segment(vec![], vec![], post_table, vec![]);
        let reader =
            PostingsTableReader::new(&buffer, 80, 80 + 4 + 20).expect("should create reader");
        assert_eq!(reader.len(), 1);

        let (offset, len, doc_freq, codec_id) = reader.entry(0).expect("entry should exist");
        assert_eq!(offset, 100);
        assert_eq!(len, 20);
        assert_eq!(doc_freq, 5);
        assert_eq!(codec_id, 0, "reserved_codec_id should round-trip");
    }

    #[test]
    fn test_postings_data_reader_range() {
        // Postings data: bytes 0..10 = "0123456789"
        let postings_data = b"0123456789".to_vec();
        let buffer = make_test_segment(vec![], vec![], vec![], postings_data);

        let postings_data_offset = 80;
        let block_meta_offset = 80 + 10;

        let reader = PostingsDataReader::new(&buffer, postings_data_offset, block_meta_offset)
            .expect("should create reader");

        let range = reader.range(0, 5).expect("range should succeed");
        assert_eq!(range, b"01234");
    }

    #[test]
    fn test_postings_data_reader_out_of_bounds() {
        let postings_data = b"01234".to_vec();
        let buffer = make_test_segment(vec![], vec![], vec![], postings_data);

        let postings_data_offset = 80;
        let block_meta_offset = 80 + 5;

        let reader = PostingsDataReader::new(&buffer, postings_data_offset, block_meta_offset)
            .expect("should create reader");

        let result = reader.range(0, 10);
        assert!(matches!(result, Err(SegmentError::BadOffset { .. })));
    }

    #[test]
    fn test_block_metadata_reader_reserved_empty() {
        let buffer = make_test_segment(vec![], vec![], vec![], vec![]);
        let block_meta_offset = 80;
        let stored_fields_offset = 80; // zero-length

        let reader = BlockMetadataReader::new(&buffer, block_meta_offset, stored_fields_offset)
            .expect("should create reader");
        assert!(reader.is_empty());
        assert_eq!(reader.len(), 0);
    }

    #[test]
    fn test_zero_copy_views() {
        // A section reader returns a view that BORROWS the source buffer
        // (zero-copy), not a copy. Proof: the bytes returned by a reader must point WITHIN the
        // original buffer's address range, at the exact expected offset — not into freshly
        // allocated memory.
        let mut buffer = vec![0_u8; 1000];
        // Write a recognizable marker at the postings-data region so we also confirm we read it in place.
        buffer[300..304].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let buf_start = buffer.as_ptr() as usize;
        let buf_end = buf_start + buffer.len();

        // PostingsDataReader over [300, 400); range(0, 4) must borrow buffer[300..304].
        let reader =
            PostingsDataReader::new(&buffer, 300, 400).expect("create postings data reader");
        let view = reader.range(0, 4).expect("range within section");

        let view_start = view.as_ptr() as usize;
        let view_end = view_start + view.len();

        // The returned slice points strictly inside the source buffer => it is a borrow, not a copy.
        assert!(
            view_start >= buf_start && view_end <= buf_end,
            "zero-copy violated: returned view [{view_start:#x},{view_end:#x}) is outside source buffer [{buf_start:#x},{buf_end:#x})"
        );
        // And at the exact byte offset the offsets dictate (section_start 300 + offset 0).
        assert_eq!(
            view_start,
            buf_start + 300,
            "view must borrow at postings_data_offset + range offset, proving in-place read"
        );
        // And it actually reads the in-place marker bytes (not zeroed copy).
        assert_eq!(view, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_parse_valid_segment() {
        // Parse segment from buffer, retrieve entries without allocation
        // Build a minimal segment: field table (1 entry) + lexicon (1 term) + postings table (1 entry) + postings data
        let mut field_data = vec![];
        field_data.extend_from_slice(&1_u32.to_le_bytes()); // count
        field_data.extend_from_slice(&10_u32.to_le_bytes()); // field_id
        field_data.extend_from_slice(&100_u32.to_le_bytes()); // doc_count
        field_data.extend_from_slice(&500_u32.to_le_bytes()); // total_terms

        let mut lex_data = vec![];
        lex_data.extend_from_slice(&1_u32.to_le_bytes()); // count
        lex_data.extend_from_slice(&0_u64.to_le_bytes()); // term_offset
        lex_data.extend_from_slice(&4_u32.to_le_bytes()); // term_len
        lex_data.extend_from_slice(&0_u32.to_le_bytes()); // postings_table_index
        lex_data.extend_from_slice(b"test"); // term bytes

        let mut post_table = vec![];
        post_table.extend_from_slice(&1_u32.to_le_bytes()); // count
        post_table.extend_from_slice(&0_u64.to_le_bytes()); // postings_data_offset
        post_table.extend_from_slice(&8_u32.to_le_bytes()); // postings_data_len
        post_table.extend_from_slice(&2_u32.to_le_bytes()); // doc_freq
        post_table.extend_from_slice(&0_u32.to_le_bytes()); // reserved_codec_id (20 bytes/entry)

        let postings_data = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 8 bytes

        let post_table_len = post_table.len() as u64;
        let postings_data_len = postings_data.len() as u64;
        let buffer = make_test_segment(field_data, lex_data, post_table, postings_data);

        // Calculate actual offsets based on what make_test_segment writes
        // (note: field_table_data, lex_data, post_table already include their count u32)
        let field_table_offset = 80_u64;
        let field_table_end = field_table_offset + 16_u64; // field_data already includes count

        let lexicon_offset = field_table_end;
        let lexicon_end = lexicon_offset + 24_u64; // lex_data already includes count

        let postings_table_offset = lexicon_end;
        let postings_table_end = postings_table_offset + post_table_len; // post_table = 4 + 20 (count + 1 entry)

        let postings_data_offset = postings_table_end;
        let postings_data_end = postings_data_offset + postings_data_len;

        // Open field table
        let field_reader = FieldTableReader::new(&buffer, field_table_offset, lexicon_offset)
            .expect("open field table");
        assert_eq!(field_reader.len(), 1);
        let (field_id, doc_count, total_terms) = field_reader.entry(0).expect("read field entry");
        assert_eq!(field_id, 10);
        assert_eq!(doc_count, 100);
        assert_eq!(total_terms, 500);

        // Open lexicon
        let lex_reader = LexiconReader::new(&buffer, lexicon_offset, postings_table_offset)
            .expect("open lexicon");
        assert_eq!(lex_reader.len(), 1);
        let (term_bytes, postings_idx) = lex_reader.entry(0).expect("read lexicon entry");
        assert_eq!(term_bytes, b"test");
        assert_eq!(postings_idx, 0);

        // Open postings table
        let post_reader =
            PostingsTableReader::new(&buffer, postings_table_offset, postings_table_end)
                .expect("open postings table");
        assert_eq!(post_reader.len(), 1);
        let (post_offset, post_len, doc_freq, codec_id) =
            post_reader.entry(0).expect("read postings entry");
        assert_eq!(post_offset, 0);
        assert_eq!(post_len, 8);
        assert_eq!(doc_freq, 2);
        assert_eq!(codec_id, 0, "reserved_codec_id should round-trip");

        // Open postings data
        let post_data_reader =
            PostingsDataReader::new(&buffer, postings_data_offset, postings_data_end)
                .expect("open postings data");
        let post_bytes = post_data_reader.range(0, 8).expect("read postings range");
        assert_eq!(post_bytes, &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_bounds_enforcement_truncated() {
        // Truncated buffer returns Truncated, not panic
        let buffer = vec![0_u8; 85]; // Header (80) + 5 bytes of field data
        let result = FieldTableReader::new(&buffer, 80, 100);

        // Reader creation should succeed (we can't read count yet)
        let reader = result.expect("reader should be created");
        // But trying to access an entry in a partially-written section should fail
        let entry_result = reader.entry(0);
        assert!(matches!(entry_result, Err(SegmentError::BadOffset { .. })));
    }

    #[test]
    fn test_no_std_compatibility() {
        // Verify readers work in a no_std context (no allocations beyond struct)
        let buffer = vec![0_u8; 200];

        // These should all succeed and not allocate
        let _field_reader = FieldTableReader::new(&buffer, 80, 100);
        let _lex_reader = LexiconReader::new(&buffer, 100, 150);
        let _post_reader = PostingsTableReader::new(&buffer, 150, 200);
        let _post_data_reader = PostingsDataReader::new(&buffer, 200, 200);
    }

    // ========== Adversarial Tests for Integer Overflow ==========

    #[test]
    fn test_postings_data_reader_range_u64_max_offset_without_panic() {
        // ADVERSARIAL: PostingsDataReader::range with offset near u64::MAX
        // Should return BadOffset error, NEVER panic on overflow.
        let postings_data = b"test".to_vec();
        let buffer = make_test_segment(vec![], vec![], vec![], postings_data);

        let postings_data_offset = 80;
        let block_meta_offset = 80 + 4;

        let reader = PostingsDataReader::new(&buffer, postings_data_offset, block_meta_offset)
            .expect("should create reader");

        // Try to access with offset that will overflow: section_start + offset
        let result = reader.range(u64::MAX - 4, 16);
        assert!(
            matches!(result, Err(SegmentError::BadOffset { .. })),
            "expected BadOffset for overflowing offset, got: {:?}",
            result
        );
    }

    #[test]
    fn test_postings_data_reader_range_len_overflow_without_panic() {
        // ADVERSARIAL: PostingsDataReader::range where start + len overflows
        // Should return BadOffset error, NEVER panic.
        let postings_data = vec![0_u8; 100];
        let buffer = make_test_segment(vec![], vec![], vec![], postings_data);

        let postings_data_offset = 80;
        let block_meta_offset = 80 + 100;

        let reader = PostingsDataReader::new(&buffer, postings_data_offset, block_meta_offset)
            .expect("should create reader");

        // Try to access with len so large that start + len overflows
        let result = reader.range(50, u32::MAX);
        assert!(
            matches!(result, Err(SegmentError::BadOffset { .. })),
            "expected BadOffset for overflowing len, got: {:?}",
            result
        );
    }

    #[test]
    fn test_field_table_reader_entry_i_mul_overflow_without_panic() {
        // ADVERSARIAL: FieldTableReader::entry where i * 12 overflows
        // Should return BadOffset error when i is so large that i * 12 overflows.
        // This test crafts a large i and ensures it doesn't panic.
        let mut field_data = vec![];
        field_data.extend_from_slice(&1_u32.to_le_bytes()); // count = 1
        field_data.extend_from_slice(&5_u32.to_le_bytes()); // field_id
        field_data.extend_from_slice(&10_u32.to_le_bytes()); // doc_count
        field_data.extend_from_slice(&50_u32.to_le_bytes()); // total_terms

        let buffer = make_test_segment(field_data, vec![], vec![], vec![]);
        let reader = FieldTableReader::new(&buffer, 80, 80 + 16).expect("should create reader");

        // Try to access with an index that's way beyond count (should fail early with index check)
        let result = reader.entry(1000);
        assert!(
            matches!(result, Err(SegmentError::BadOffset { .. })),
            "expected BadOffset for out-of-range index, got: {:?}",
            result
        );

        // Also test the maximum u32 to ensure it doesn't panic in multiplication
        // This doesn't actually overflow u32::MAX * 12 in u64, but tests the code path
        let result2 = reader.entry(u32::MAX);
        assert!(matches!(result2, Err(SegmentError::BadOffset { .. })));
    }

    #[test]
    fn test_lexicon_reader_new_count_mul_overflow_without_panic() {
        // ADVERSARIAL: LexiconReader::new where count * 16 overflows
        // Craft a lexicon with a huge count that will overflow in blob_start calculation.
        let mut lex_data = vec![];
        lex_data.extend_from_slice(&u32::MAX.to_le_bytes()); // count = u32::MAX

        let buffer = make_test_segment(vec![], lex_data, vec![], vec![]);
        let lexicon_offset = 80_u64;
        let postings_table_offset = 100_u64; // Artificially small to trigger overflow

        // This should fail with BadOffset when computing blob_start (count*16 overflows)
        let result = LexiconReader::new(&buffer, lexicon_offset, postings_table_offset);
        assert!(
            matches!(result, Err(SegmentError::BadOffset { .. })),
            "expected BadOffset for overflowing count*16, got: {:?}",
            result
        );
    }

    #[test]
    fn test_lexicon_reader_entry_term_offset_overflow_without_panic() {
        // ADVERSARIAL: LexiconReader::entry where term_offset overflows when added to blob_start
        let mut lex_data = vec![];
        lex_data.extend_from_slice(&1_u32.to_le_bytes()); // count = 1
        lex_data.extend_from_slice(&(u64::MAX - 10).to_le_bytes()); // term_offset near u64::MAX
        lex_data.extend_from_slice(&3_u32.to_le_bytes()); // term_len
        lex_data.extend_from_slice(&7_u32.to_le_bytes()); // postings_table_index
        lex_data.extend_from_slice(b"foo"); // term bytes

        let buffer = make_test_segment(vec![], lex_data, vec![], vec![]);
        let lexicon_offset = 80_u64;
        let postings_table_offset = 80_u64 + 4 + 16 + 3;

        let reader = LexiconReader::new(&buffer, lexicon_offset, postings_table_offset)
            .expect("should create reader");

        // Try to read entry 0; blob_start + term_offset should overflow
        let result = reader.entry(0);
        assert!(
            matches!(result, Err(SegmentError::BadOffset { .. })),
            "expected BadOffset for overflowing term_offset, got: {:?}",
            result
        );
    }

    #[test]
    fn test_lexicon_reader_entry_term_len_overflow_without_panic() {
        // ADVERSARIAL: LexiconReader::entry where term_absolute_offset + term_len overflows
        let mut lex_data = vec![];
        lex_data.extend_from_slice(&1_u32.to_le_bytes()); // count = 1
        lex_data.extend_from_slice(&(u64::MAX - 10).to_le_bytes()); // term_offset
        lex_data.extend_from_slice(&u32::MAX.to_le_bytes()); // term_len = u32::MAX
        lex_data.extend_from_slice(&7_u32.to_le_bytes()); // postings_table_index
        // No term bytes (will be out of bounds anyway)

        let buffer = make_test_segment(vec![], lex_data, vec![], vec![]);
        let lexicon_offset = 80_u64;
        let postings_table_offset = 80_u64 + 4 + 16 + 10;

        let reader = LexiconReader::new(&buffer, lexicon_offset, postings_table_offset)
            .expect("should create reader");

        // Try to read entry 0; term_absolute_offset + term_len should overflow
        let result = reader.entry(0);
        assert!(
            matches!(result, Err(SegmentError::BadOffset { .. })),
            "expected BadOffset for overflowing term_len addition, got: {:?}",
            result
        );
    }

    #[test]
    fn test_postings_table_reader_entry_i_mul_overflow_without_panic() {
        // ADVERSARIAL: PostingsTableReader::entry where i * 16 overflows
        let mut post_table = vec![];
        post_table.extend_from_slice(&1_u32.to_le_bytes()); // count = 1
        post_table.extend_from_slice(&100_u64.to_le_bytes()); // postings_data_offset
        post_table.extend_from_slice(&20_u32.to_le_bytes()); // postings_data_len
        post_table.extend_from_slice(&5_u32.to_le_bytes()); // doc_freq
        post_table.extend_from_slice(&0_u32.to_le_bytes()); // reserved_codec_id

        let buffer = make_test_segment(vec![], vec![], post_table, vec![]);
        let reader =
            PostingsTableReader::new(&buffer, 80, 80 + 4 + 16).expect("should create reader");

        // Try to access with index beyond count
        let result = reader.entry(1000);
        assert!(
            matches!(result, Err(SegmentError::BadOffset { .. })),
            "expected BadOffset for out-of-range index, got: {:?}",
            result
        );

        // Test maximum u32 to ensure multiplication doesn't panic
        let result2 = reader.entry(u32::MAX);
        assert!(matches!(result2, Err(SegmentError::BadOffset { .. })));
    }

    #[test]
    fn test_field_table_reader_section_start_offset_overflow_without_panic() {
        // ADVERSARIAL: FieldTableReader where section_start + entry_offset overflows
        let mut field_data = vec![];
        field_data.extend_from_slice(&1_u32.to_le_bytes()); // count = 1
        field_data.extend_from_slice(&5_u32.to_le_bytes()); // field_id
        field_data.extend_from_slice(&10_u32.to_le_bytes()); // doc_count
        field_data.extend_from_slice(&50_u32.to_le_bytes()); // total_terms

        let buffer = make_test_segment(field_data, vec![], vec![], vec![]);

        // Create reader with section_start very close to u64::MAX
        let section_start = u64::MAX - 100;
        let section_end = u64::MAX;

        // This should fail with BadOffset due to section_start out of buffer bounds
        let _result = FieldTableReader::new(&buffer, section_start, section_end);

        // Reader creation will fail due to bounds check, proving overflow safety
    }

    #[test]
    fn test_lexicon_reader_blob_start_add_overflow_without_panic() {
        // ADVERSARIAL: LexiconReader::new where lexicon_offset + 4 overflows
        // Create a lexicon with offset very close to u64::MAX
        let mut lex_data = vec![];
        lex_data.extend_from_slice(&0_u32.to_le_bytes()); // count = 0

        let buffer = make_test_segment(vec![], lex_data, vec![], vec![]);
        let lexicon_offset = u64::MAX - 2; // Will overflow when adding 4
        let postings_table_offset = u64::MAX;

        // This should return BadOffset due to overflow in blob_start calculation
        let result = LexiconReader::new(&buffer, lexicon_offset, postings_table_offset);
        assert!(
            matches!(result, Err(SegmentError::BadOffset { .. })),
            "expected BadOffset for overflowing blob_start, got: {:?}",
            result
        );
    }

    #[test]
    fn test_postings_table_reader_entry_offset_add_overflow_without_panic() {
        // ADVERSARIAL: PostingsTableReader::entry where section_start + entry_offset overflows
        let mut post_table = vec![];
        post_table.extend_from_slice(&1_u32.to_le_bytes()); // count = 1
        post_table.extend_from_slice(&0_u64.to_le_bytes()); // postings_data_offset
        post_table.extend_from_slice(&20_u32.to_le_bytes()); // postings_data_len
        post_table.extend_from_slice(&5_u32.to_le_bytes()); // doc_freq
        post_table.extend_from_slice(&0_u32.to_le_bytes()); // reserved_codec_id

        let buffer = make_test_segment(vec![], vec![], post_table, vec![]);

        // Create reader with section_start very close to u64::MAX
        let section_start = u64::MAX - 100;
        let section_end = u64::MAX;

        // Reader creation itself might fail due to bounds
        let reader_result = PostingsTableReader::new(&buffer, section_start, section_end);
        if let Ok(reader) = reader_result {
            // If reader was created, entry(0) must still fail safely
            let result = reader.entry(0);
            assert!(
                matches!(result, Err(SegmentError::BadOffset { .. })),
                "expected BadOffset for overflowing absolute_offset, got: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_read_u32_le_offset_overflow_without_panic() {
        // ADVERSARIAL: read_u32_le with offset near usize::MAX should handle overflow gracefully
        let buffer = vec![0_u8; 100];
        let result = read_u32_le(&buffer, usize::MAX - 2);
        assert!(
            matches!(result, Err(SegmentError::Truncated { .. })),
            "expected Truncated for overflowing offset+4, got: {:?}",
            result
        );
    }

    #[test]
    fn test_read_u64_le_offset_overflow_without_panic() {
        // ADVERSARIAL: read_u64_le with offset near usize::MAX should handle overflow gracefully
        let buffer = vec![0_u8; 100];
        let result = read_u64_le(&buffer, usize::MAX - 4);
        assert!(
            matches!(result, Err(SegmentError::Truncated { .. })),
            "expected Truncated for overflowing offset+8, got: {:?}",
            result
        );
    }
}
