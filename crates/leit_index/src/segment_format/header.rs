// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Segment header: fixed-layout, little-endian POD.
//!
//! The header is exactly 80 bytes and sits at offset 0 of every segment:
//! - `magic`: u32 LE (4 bytes)
//! - `version`: u32 LE (4 bytes)
//! - `format_flags`: u32 LE (4 bytes)
//! - `document_count`: u32 LE (4 bytes, total number of documents in this segment)
//! - `field_table_offset`: u64 LE (8 bytes)
//! - `lexicon_offset`: u64 LE (8 bytes)
//! - `postings_table_offset`: u64 LE (8 bytes)
//! - `postings_data_offset`: u64 LE (8 bytes)
//! - `block_meta_offset`: u64 LE (8 bytes)
//! - `stored_fields_offset`: u64 LE (8 bytes)
//! - `columnar_offset`: u64 LE (8 bytes)
//! - `footer_offset`: u64 LE (8 bytes)
//!
//! Total: 4 + 4 + 4 + 4 + (8 * 8) = 16 + 64 = 80 bytes.
//!
//! This layout uses manual (not bytemuck `Pod`) serialization to ensure
//! explicit control over the byte structure and avoid padding issues.

use crate::error::SegmentError;

/// Magic constant identifying the v1 segment format.
///
/// Deliberately DISTINCT from the legacy directory format's magic (`b"LSEG"`,
/// `0x4745534C`): a legacy segment fed to this reader must hit [`SegmentError::BadMagic`]
/// (clean rejection → rebuild, never silent corruption), not be
/// misread as a v1 header. Value is `b"LSG1"` (leit segment, format 1) in LE.
pub const MAGIC: u32 = u32::from_le_bytes(*b"LSG1");

/// Format version for v1 segment header.
pub const FORMAT_VERSION: u32 = 1;

/// Fixed serialized header size in bytes.
pub const HEADER_SIZE: usize = 80;

/// Segment header: fixed-layout, little-endian POD.
///
/// The header occupies exactly 80 bytes at offset 0 of every segment.
/// All offsets are absolute from segment start (little-endian u64).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentHeader {
    /// 4-byte magic constant for format identification.
    pub magic: u32,
    /// Format version (current: 1).
    pub version: u32,
    /// Feature/optional-section flags.
    pub format_flags: u32,
    /// Total number of documents in this segment.
    pub document_count: u32,
    /// Absolute offset to field table section (or next section if `field_table_offset` is zero-length).
    pub field_table_offset: u64,
    /// Absolute offset to lexicon section (or next section if `lexicon_offset` is zero-length).
    pub lexicon_offset: u64,
    /// Absolute offset to postings metadata table (or next section if zero-length).
    pub postings_table_offset: u64,
    /// Absolute offset to postings data blocks (or next section if zero-length).
    pub postings_data_offset: u64,
    /// Absolute offset to block metadata section (reserved, may be zero-length in minimal v1).
    pub block_meta_offset: u64,
    /// Absolute offset to stored fields section (optional, reserved for future extensions).
    pub stored_fields_offset: u64,
    /// Absolute offset to columnar section (optional, reserved for future extensions).
    pub columnar_offset: u64,
    /// Absolute offset to footer (checksum, reserved, may be zero-length in minimal v1).
    pub footer_offset: u64,
}

impl SegmentHeader {
    /// Encode header to a fixed 80-byte buffer in little-endian format.
    ///
    /// # Returns
    /// A 80-byte array with the serialized header.
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0_u8; HEADER_SIZE];

        // Offsets 0-3: magic (u32 LE)
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());

        // Offsets 4-7: version (u32 LE)
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());

        // Offsets 8-11: format_flags (u32 LE)
        buf[8..12].copy_from_slice(&self.format_flags.to_le_bytes());

        // Offsets 12-15: document_count (u32 LE)
        buf[12..16].copy_from_slice(&self.document_count.to_le_bytes());

        // Offsets 16-23: field_table_offset (u64 LE)
        buf[16..24].copy_from_slice(&self.field_table_offset.to_le_bytes());

        // Offsets 24-31: lexicon_offset (u64 LE)
        buf[24..32].copy_from_slice(&self.lexicon_offset.to_le_bytes());

        // Offsets 32-39: postings_table_offset (u64 LE)
        buf[32..40].copy_from_slice(&self.postings_table_offset.to_le_bytes());

        // Offsets 40-47: postings_data_offset (u64 LE)
        buf[40..48].copy_from_slice(&self.postings_data_offset.to_le_bytes());

        // Offsets 48-55: block_meta_offset (u64 LE)
        buf[48..56].copy_from_slice(&self.block_meta_offset.to_le_bytes());

        // Offsets 56-63: stored_fields_offset (u64 LE)
        buf[56..64].copy_from_slice(&self.stored_fields_offset.to_le_bytes());

        // Offsets 64-71: columnar_offset (u64 LE)
        buf[64..72].copy_from_slice(&self.columnar_offset.to_le_bytes());

        // Offsets 72-79: footer_offset (u64 LE)
        buf[72..80].copy_from_slice(&self.footer_offset.to_le_bytes());

        buf
    }

    /// Decode header from a buffer.
    ///
    /// # Arguments
    /// * `bytes` - buffer to read from; must be at least `HEADER_SIZE` bytes
    ///
    /// # Returns
    /// Decoded `SegmentHeader` or `SegmentError` if validation fails.
    ///
    /// # Errors
    /// - `SegmentError::Truncated` if buffer is shorter than `HEADER_SIZE`
    /// - `SegmentError::BadMagic{found}` if magic field does not match `MAGIC` constant
    /// - `SegmentError::UnsupportedVersion{found, expected}` if version != `FORMAT_VERSION`
    pub fn read(bytes: &[u8]) -> Result<Self, SegmentError> {
        // Check minimum length
        if bytes.len() < HEADER_SIZE {
            return Err(SegmentError::Truncated {
                needed: HEADER_SIZE,
                found: bytes.len(),
            });
        }

        // Read magic at offset 0-3
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        // Check magic
        if magic != MAGIC {
            return Err(SegmentError::BadMagic { found: magic });
        }

        // Read version at offset 4-7
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        // Check version
        if version != FORMAT_VERSION {
            return Err(SegmentError::UnsupportedVersion {
                found: version,
                expected: FORMAT_VERSION,
            });
        }

        // Read format_flags at offset 8-11
        let format_flags = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        // Read document_count at offset 12-15
        let document_count = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        // Read field_table_offset at offset 16-23
        let field_table_offset = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);

        // Read lexicon_offset at offset 24-31
        let lexicon_offset = u64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);

        // Read postings_table_offset at offset 32-39
        let postings_table_offset = u64::from_le_bytes([
            bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37], bytes[38], bytes[39],
        ]);

        // Read postings_data_offset at offset 40-47
        let postings_data_offset = u64::from_le_bytes([
            bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
        ]);

        // Read block_meta_offset at offset 48-55
        let block_meta_offset = u64::from_le_bytes([
            bytes[48], bytes[49], bytes[50], bytes[51], bytes[52], bytes[53], bytes[54], bytes[55],
        ]);

        // Read stored_fields_offset at offset 56-63
        let stored_fields_offset = u64::from_le_bytes([
            bytes[56], bytes[57], bytes[58], bytes[59], bytes[60], bytes[61], bytes[62], bytes[63],
        ]);

        // Read columnar_offset at offset 64-71
        let columnar_offset = u64::from_le_bytes([
            bytes[64], bytes[65], bytes[66], bytes[67], bytes[68], bytes[69], bytes[70], bytes[71],
        ]);

        // Read footer_offset at offset 72-79
        let footer_offset = u64::from_le_bytes([
            bytes[72], bytes[73], bytes[74], bytes[75], bytes[76], bytes[77], bytes[78], bytes[79],
        ]);

        Ok(Self {
            magic,
            version,
            format_flags,
            document_count,
            field_table_offset,
            lexicon_offset,
            postings_table_offset,
            postings_data_offset,
            block_meta_offset,
            stored_fields_offset,
            columnar_offset,
            footer_offset,
        })
    }

    /// Validate section layout: offsets in-bounds, ordered, with reserved zero-length sections allowed.
    ///
    /// Rules:
    /// - All offsets must be <= `buffer_len` (no offset beyond segment)
    /// - Sections must be ordered: `field_table` <= `lexicon` <= `postings_table` <= `postings_data`
    /// - A zero-length section shares its offset with the next (`offset_n == offset_{n+1}` is valid)
    /// - Sections are either populated (strict <) or reserved (==); no overlap
    /// - `footer_offset` terminates the segment
    ///
    /// # Arguments
    /// * `buffer_len` - total segment buffer size
    ///
    /// # Returns
    /// `Ok(())` if layout is valid, or `SegmentError::BadSectionLayout` / `BadOffset`
    pub fn validate_layout(&self, buffer_len: u64) -> Result<(), SegmentError> {
        // All offsets must be within buffer
        if self.field_table_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: self.field_table_offset,
                limit: buffer_len,
            });
        }
        if self.lexicon_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: self.lexicon_offset,
                limit: buffer_len,
            });
        }
        if self.postings_table_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: self.postings_table_offset,
                limit: buffer_len,
            });
        }
        if self.postings_data_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: self.postings_data_offset,
                limit: buffer_len,
            });
        }
        if self.block_meta_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: self.block_meta_offset,
                limit: buffer_len,
            });
        }
        if self.stored_fields_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: self.stored_fields_offset,
                limit: buffer_len,
            });
        }
        if self.columnar_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: self.columnar_offset,
                limit: buffer_len,
            });
        }
        if self.footer_offset > buffer_len {
            return Err(SegmentError::BadOffset {
                offset: self.footer_offset,
                limit: buffer_len,
            });
        }

        // Sections must be ordered: field_table <= lexicon <= postings_table <= postings_data
        // <= block_meta <= stored_fields <= columnar <= footer
        // The rule: if two sections are not at the same offset (one is zero-length), they must be strictly ordered.
        // Equivalently: if section_a is zero-length (next_offset == its_offset), it's OK;
        // if section_a is populated (next_offset > its_offset), then section_b's offset must be >= next_offset.
        // For clarity, we apply the rule: offset_n <= offset_{n+1} for all adjacent pairs,
        // and this covers both populated and zero-length cases.

        if self.field_table_offset > self.lexicon_offset {
            return Err(SegmentError::BadSectionLayout);
        }
        if self.lexicon_offset > self.postings_table_offset {
            return Err(SegmentError::BadSectionLayout);
        }
        if self.postings_table_offset > self.postings_data_offset {
            return Err(SegmentError::BadSectionLayout);
        }
        if self.postings_data_offset > self.block_meta_offset {
            return Err(SegmentError::BadSectionLayout);
        }
        if self.block_meta_offset > self.stored_fields_offset {
            return Err(SegmentError::BadSectionLayout);
        }
        if self.stored_fields_offset > self.columnar_offset {
            return Err(SegmentError::BadSectionLayout);
        }
        if self.columnar_offset > self.footer_offset {
            return Err(SegmentError::BadSectionLayout);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_encode_decode_roundtrip() {
        let original = SegmentHeader {
            magic: MAGIC,
            version: FORMAT_VERSION,
            format_flags: 0,
            document_count: 0,
            field_table_offset: 80,
            lexicon_offset: 200,
            postings_table_offset: 400,
            postings_data_offset: 600,
            block_meta_offset: 1000,
            stored_fields_offset: 1000,
            columnar_offset: 1000,
            footer_offset: 1200,
        };

        let encoded = original.encode();
        assert_eq!(encoded.len(), HEADER_SIZE);

        let decoded = SegmentHeader::read(&encoded).expect("decode should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_header_encode_produces_80_bytes() {
        let header = SegmentHeader {
            magic: MAGIC,
            version: FORMAT_VERSION,
            format_flags: 0,
            document_count: 0,
            field_table_offset: 80,
            lexicon_offset: 200,
            postings_table_offset: 400,
            postings_data_offset: 600,
            block_meta_offset: 1000,
            stored_fields_offset: 1000,
            columnar_offset: 1000,
            footer_offset: 1200,
        };

        let encoded = header.encode();
        assert_eq!(encoded.len(), 80);
    }

    #[test]
    fn test_header_rejects_truncated_buffer() {
        let short_buf = [0_u8; 79];
        let result = SegmentHeader::read(&short_buf);
        assert!(matches!(
            result,
            Err(SegmentError::Truncated {
                needed: 80,
                found: 79
            })
        ));
    }

    #[test]
    fn test_header_rejects_bad_magic() {
        let mut buf = [0_u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&0xDEADBEEF_u32.to_le_bytes());
        buf[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());

        let result = SegmentHeader::read(&buf);
        assert!(matches!(
            result,
            Err(SegmentError::BadMagic { found: 0xDEADBEEF })
        ));
    }

    #[test]
    fn rejects_legacy_lseg_magic() {
        // A legacy segment begins with b"LSEG" and version=1 (which decodes to
        // u32 1 in the new header's version slot). With a DISTINCT v1 magic, such a buffer
        // is cleanly rejected at the magic check (never silent corruption)
        // rather than being misread as a v1 header.
        let legacy_magic = u32::from_le_bytes(*b"LSEG");
        let mut buf = [0_u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&legacy_magic.to_le_bytes());
        buf[4..8].copy_from_slice(&1_u32.to_le_bytes());

        let result = SegmentHeader::read(&buf);
        assert!(matches!(
            result,
            Err(SegmentError::BadMagic { found }) if found == legacy_magic
        ));
    }

    #[test]
    fn test_header_rejects_unsupported_version() {
        let mut buf = [0_u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&99_u32.to_le_bytes());

        let result = SegmentHeader::read(&buf);
        assert!(matches!(
            result,
            Err(SegmentError::UnsupportedVersion {
                found: 99,
                expected: 1
            })
        ));
    }

    #[test]
    fn test_validate_layout_rejects_offset_out_of_bounds() {
        let header = SegmentHeader {
            magic: MAGIC,
            version: FORMAT_VERSION,
            format_flags: 0,
            document_count: 0,
            field_table_offset: 80,
            lexicon_offset: 200,
            postings_table_offset: 400,
            postings_data_offset: 600,
            block_meta_offset: 1000,
            stored_fields_offset: 1000,
            columnar_offset: 1000,
            footer_offset: 2000, // Exceeds buffer
        };

        let result = header.validate_layout(1500);
        assert!(matches!(
            result,
            Err(SegmentError::BadOffset {
                offset: 2000,
                limit: 1500
            })
        ));
    }

    #[test]
    fn test_validate_layout_rejects_unordered_sections() {
        let header = SegmentHeader {
            magic: MAGIC,
            version: FORMAT_VERSION,
            format_flags: 0,
            document_count: 0,
            field_table_offset: 200, // Out of order
            lexicon_offset: 80,
            postings_table_offset: 400,
            postings_data_offset: 600,
            block_meta_offset: 1000,
            stored_fields_offset: 1000,
            columnar_offset: 1000,
            footer_offset: 1200,
        };

        let result = header.validate_layout(2000);
        assert!(matches!(result, Err(SegmentError::BadSectionLayout)));
    }

    #[test]
    fn test_validate_layout_accepts_monotonic_offsets() {
        let header = SegmentHeader {
            magic: MAGIC,
            version: FORMAT_VERSION,
            format_flags: 0,
            document_count: 0,
            field_table_offset: 80,
            lexicon_offset: 200,
            postings_table_offset: 400,
            postings_data_offset: 600,
            block_meta_offset: 1000,
            stored_fields_offset: 1000,
            columnar_offset: 1000,
            footer_offset: 1200,
        };

        assert!(header.validate_layout(2000).is_ok());
    }

    #[test]
    fn test_validate_layout_accepts_zero_length_sections() {
        // Zero-length section: offset_n == offset_{n+1}
        let header = SegmentHeader {
            magic: MAGIC,
            version: FORMAT_VERSION,
            format_flags: 0,
            document_count: 0,
            field_table_offset: 80,
            lexicon_offset: 200,
            postings_table_offset: 200, // Zero-length (== previous)
            postings_data_offset: 200,
            block_meta_offset: 1000,
            stored_fields_offset: 1000,
            columnar_offset: 1000,
            footer_offset: 1200,
        };

        assert!(header.validate_layout(2000).is_ok());
    }

    #[test]
    fn test_validate_layout_rejects_overlap() {
        // Simulate overlap: stored_fields < columnar but they should be ordered
        let header = SegmentHeader {
            magic: MAGIC,
            version: FORMAT_VERSION,
            format_flags: 0,
            document_count: 0,
            field_table_offset: 80,
            lexicon_offset: 200,
            postings_table_offset: 400,
            postings_data_offset: 600,
            block_meta_offset: 1000,
            stored_fields_offset: 1200,
            columnar_offset: 1100, // Out of order
            footer_offset: 1300,
        };

        let result = header.validate_layout(2000);
        assert!(matches!(result, Err(SegmentError::BadSectionLayout)));
    }

    #[test]
    fn rejects_unknown_version() {
        // Reader cleanly rejects segment with unknown version
        let mut buf = [0_u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&99_u32.to_le_bytes()); // Version 99

        let result = SegmentHeader::read(&buf);
        match result {
            Err(SegmentError::UnsupportedVersion { found, expected }) => {
                assert_eq!(found, 99);
                assert_eq!(expected, FORMAT_VERSION);
            }
            _ => panic!("Expected UnsupportedVersion error"),
        }
    }
}
