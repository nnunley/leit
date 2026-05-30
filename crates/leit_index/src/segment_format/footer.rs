// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Segment footer: fixed-layout, little-endian POD carrying a CRC32C checksum (STORY-0029, STORY-0047).
//!
//! The footer is a fixed-size structure located at `footer_offset` from the segment start.
//! It contains:
//! - `checksum`: `u32` LE (4 bytes) — CRC32C over the covered byte range `[0, footer_offset)`
//!
//! Total: 4 bytes.
//!
//! **Checksum coverage (STORY-0029 AC-2):** The checksum value protects all segment bytes
//! from offset 0 up to (but not including) the footer start (`footer_offset`). This includes
//! the header, field table, lexicon, postings table, postings data, and block metadata sections.
//! The footer itself is NOT included in the checksum (the checksum cannot include its own value).
//!
//! **Checksum algorithm:** CRC32C (Castagnoli polynomial 0x1EDC6F41, also known as ISCSI CRC).
//! Chosen for:
//! - Deterministic, reproducible across platforms (unlike hardware-dependent CRC)
//! - `no_std` compatible (implemented inline, no external dependencies)
//! - Fast bitwise computation
//! - Catches bit-flips and byte corruption effectively
//!
//! **Validation (DEC-07 + STORY-0026 AC-6):** Footer verification is performed only in
//! `ValidationMode::Full`. The `Footer::verify()` function reads the footer checksum,
//! recomputes the checksum over the covered bytes, and returns `BadChecksum` if they
//! do not match. This primitive is called by `open_with_validation(Full)` during segment open
//! (implemented in ITER-0004 T6).

use crate::error::SegmentError;

/// Fixed serialized footer size in bytes.
pub const FOOTER_SIZE: usize = 4;

/// Segment footer: fixed-layout, little-endian POD (STORY-0029).
///
/// The footer occupies exactly 4 bytes at offset `footer_offset` of every segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footer {
    /// CRC32C checksum over bytes `[0, footer_offset)`.
    pub checksum: u32,
}

impl Footer {
    /// Encode footer to a fixed 4-byte buffer in little-endian format.
    ///
    /// # Returns
    /// A 4-byte array with the serialized checksum.
    pub fn encode(&self) -> [u8; FOOTER_SIZE] {
        let mut buf = [0_u8; FOOTER_SIZE];
        buf[0..4].copy_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    /// Decode footer from a buffer.
    ///
    /// # Arguments
    /// * `bytes` - buffer to read from; must be at least `FOOTER_SIZE` bytes
    ///
    /// # Returns
    /// Decoded `Footer` or `SegmentError::Truncated` if buffer is too short.
    pub fn read(bytes: &[u8]) -> Result<Self, SegmentError> {
        if bytes.len() < FOOTER_SIZE {
            return Err(SegmentError::Truncated {
                needed: FOOTER_SIZE,
                found: bytes.len(),
            });
        }

        let checksum = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        Ok(Self { checksum })
    }

    /// Verify footer checksum against segment buffer.
    ///
    /// Reads the footer at `footer_offset`, recomputes the checksum over the covered
    /// byte range `[0, footer_offset)`, and returns `BadChecksum` if they do not match.
    /// This is the Full-mode validation entry point (STORY-0026 AC-6).
    ///
    /// # Arguments
    /// * `segment_bytes` - complete segment buffer
    /// * `footer_offset` - absolute offset to footer start
    ///
    /// # Returns
    /// `Ok(())` if checksum matches, or `SegmentError::BadChecksum` if corrupted.
    /// Returns `SegmentError::Truncated` if footer is not fully readable.
    /// Returns `SegmentError::BadOffset` if `footer_offset` is out of bounds.
    pub fn verify(segment_bytes: &[u8], footer_offset: u64) -> Result<(), SegmentError> {
        // Validate footer_offset is in bounds
        let footer_offset_usize = usize::try_from(footer_offset).unwrap_or(usize::MAX);
        if footer_offset_usize > segment_bytes.len() {
            return Err(SegmentError::BadOffset {
                offset: footer_offset,
                limit: segment_bytes.len() as u64,
            });
        }

        // Ensure footer is fully readable
        if segment_bytes.len() < footer_offset_usize + FOOTER_SIZE {
            return Err(SegmentError::Truncated {
                needed: footer_offset_usize + FOOTER_SIZE,
                found: segment_bytes.len(),
            });
        }

        // Read footer from buffer
        let footer_bytes = &segment_bytes[footer_offset_usize..footer_offset_usize + FOOTER_SIZE];
        let footer = Self::read(footer_bytes)?;

        // Recompute checksum over covered bytes [0, footer_offset)
        let covered = &segment_bytes[..footer_offset_usize];
        let computed = compute_checksum(covered);

        // Compare
        if computed != footer.checksum {
            return Err(SegmentError::BadChecksum {
                expected: footer.checksum,
                found: computed,
            });
        }

        Ok(())
    }
}

/// Compute CRC32C (Castagnoli) checksum over a byte range.
///
/// Uses the polynomial `0x1EDC6F41`. Deterministic and reproducible across all platforms.
/// This is the same CRC32C used in iSCSI and many other protocols.
///
/// # Arguments
/// * `data` - bytes to checksum
///
/// # Returns
/// CRC32C checksum as `u32` (little-endian stored in segment).
pub fn compute_checksum(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;

    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0x82F63B78; // Reversed polynomial for right-shift variant
            } else {
                crc >>= 1;
            }
        }
    }

    crc ^ 0xFFFFFFFF
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_footer_encode_decode_roundtrip() {
        let original = Footer {
            checksum: 0xDEADBEEF,
        };

        let encoded = original.encode();
        assert_eq!(encoded.len(), FOOTER_SIZE);

        let decoded = Footer::read(&encoded).expect("decode should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_footer_encode_produces_4_bytes() {
        let footer = Footer {
            checksum: 0x12345678,
        };
        let encoded = footer.encode();
        assert_eq!(encoded.len(), 4);
    }

    #[test]
    fn test_footer_rejects_truncated_buffer() {
        let short_buf = [0_u8; 3];
        let result = Footer::read(&short_buf);
        assert!(matches!(
            result,
            Err(SegmentError::Truncated {
                needed: 4,
                found: 3
            })
        ));
    }

    #[test]
    fn test_checksum_empty_buffer() {
        let empty = [];
        let crc = compute_checksum(&empty);
        // Empty buffer should produce 0 for CRC32C with the final XOR
        assert_eq!(crc, 0);
    }

    #[test]
    fn test_checksum_deterministic() {
        let data = b"hello world";
        let crc1 = compute_checksum(data);
        let crc2 = compute_checksum(data);
        assert_eq!(crc1, crc2);
    }

    #[test]
    fn test_verify_untampered_footer() {
        // Build a segment with valid footer
        let mut segment = vec![0_u8; 100];
        let covered = &segment[..80]; // Cover bytes 0..80
        let checksum = compute_checksum(covered);

        // Write footer at offset 80
        let footer = Footer { checksum };
        let footer_bytes = footer.encode();
        segment[80..84].copy_from_slice(&footer_bytes);

        // Verify should succeed
        let result = Footer::verify(&segment, 80);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_detects_corruption() {
        // Build a segment with valid footer
        let mut segment = vec![0_u8; 100];
        let covered = &segment[..80]; // Cover bytes 0..80
        let checksum = compute_checksum(covered);

        // Write footer at offset 80
        let footer = Footer { checksum };
        let footer_bytes = footer.encode();
        segment[80..84].copy_from_slice(&footer_bytes);

        // Corrupt one byte in the covered range
        segment[40] ^= 0xFF;

        // Verify should now fail with BadChecksum
        let result = Footer::verify(&segment, 80);
        assert!(matches!(result, Err(SegmentError::BadChecksum { .. })));

        // Check that the error shows the expected vs found checksums
        if let Err(SegmentError::BadChecksum { expected, found }) = result {
            assert_eq!(expected, checksum);
            assert_ne!(found, checksum);
        } else {
            panic!("expected BadChecksum error");
        }
    }

    #[test]
    fn test_verify_footer_offset_out_of_bounds() {
        let segment = vec![0_u8; 50];
        let result = Footer::verify(&segment, 100);
        assert!(matches!(
            result,
            Err(SegmentError::BadOffset { offset: 100, .. })
        ));
    }

    #[test]
    fn test_verify_footer_truncated() {
        let segment = vec![0_u8; 82]; // Only 82 bytes, footer at 80 needs 4 bytes (80-83)
        let result = Footer::verify(&segment, 80);
        assert!(matches!(
            result,
            Err(SegmentError::Truncated {
                needed: 84,
                found: 82
            })
        ));
    }

    #[test]
    fn test_checksum_different_data_different_crc() {
        let data1 = b"hello";
        let data2 = b"world";
        let crc1 = compute_checksum(data1);
        let crc2 = compute_checksum(data2);
        assert_ne!(crc1, crc2);
    }

    #[test]
    fn test_verify_with_multiple_sections() {
        // Simulate a segment with header (80 bytes) + field_table (100 bytes) + footer
        let mut segment = vec![0_u8; 184]; // 80 + 100 + 4
        let footer_offset = 180;

        // Compute checksum over everything except footer
        let covered = &segment[..footer_offset];
        let checksum = compute_checksum(covered);

        // Write footer
        let footer = Footer { checksum };
        let footer_bytes = footer.encode();
        segment[footer_offset..footer_offset + FOOTER_SIZE].copy_from_slice(&footer_bytes);

        // Verify should succeed
        let result = Footer::verify(&segment, footer_offset as u64);
        assert!(result.is_ok());

        // Corrupt in the field_table section
        segment[150] ^= 0xFF;

        // Verify should now fail
        let result = Footer::verify(&segment, footer_offset as u64);
        assert!(matches!(result, Err(SegmentError::BadChecksum { .. })));
    }
}
