// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compressed postings codecs for efficient storage and traversal.
//!
//! This module provides codec implementations for encoding and decoding postings lists.
//! Multiple codec strategies are supported to balance decode cost and memory footprint:
//!
//! - **`DeltaVarint`**: Single-block encoding using delta-encoded doc IDs and varint-encoded TFs.
//! - **`BlockDelta`**: Multi-block encoding with 128-doc blocks, each independently decodable.
//!
//! ## Codec ID marker
//!
//! Encoded postings are prefixed by a 1-byte `CodecId` to support multiple codec implementations.
//! See DEC-12 in the architectural decisions for the full specification.

use alloc::vec::Vec;
use core::fmt;
use leit_core::{SegmentLocalDocId, TermFreq};

/// Fixed block size for `BlockDelta` codec: 128 documents per block.
///
/// This constant is defined in one place (DEC-11) so that a future codec may tune it
/// without a format break. The block count is encoded per block, not assumed by readers.
pub const BLOCK_DOC_COUNT: usize = 128;

/// Codec identifier for selecting among multiple postings codec implementations.
///
/// Each postings list is prefixed by a single byte codec marker (1-byte `CodecId`).
/// Decoders read this marker to dispatch to the correct codec implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum CodecId {
    /// Delta-encoded doc IDs + varint-encoded TFs in a single stream (no block structure).
    DeltaVarint = 0,
    /// Block-based codec: 128-doc blocks, each independently decodable.
    BlockDelta = 1,
}

impl CodecId {
    /// Convert a byte to a `CodecId`, returning `None` if the byte is not a valid marker.
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::DeltaVarint),
            1 => Some(Self::BlockDelta),
            _ => None,
        }
    }

    /// Convert this `CodecId` to a byte.
    pub const fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Errors that can occur during codec operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// The byte stream is truncated or incomplete.
    Truncated,
    /// The codec ID marker is not recognized.
    BadMarker(u8),
    /// Invalid block count in block header.
    InvalidBlockCount,
    /// Invalid varint encoding.
    InvalidVarint,
    /// A block header's `first_doc`/`last_doc` range does not match its decoded doc stream.
    BlockHeaderMismatch,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "byte stream truncated"),
            Self::BadMarker(byte) => write!(f, "unrecognized codec ID: {byte}"),
            Self::InvalidBlockCount => write!(f, "invalid block count"),
            Self::InvalidVarint => write!(f, "invalid varint encoding"),
            Self::BlockHeaderMismatch => {
                write!(f, "block header doc-range does not match decoded stream")
            }
        }
    }
}

#[cfg(feature = "std")]
impl core::error::Error for CodecError {}

/// Core codec interface for encoding and decoding postings.
///
/// A codec is responsible for:
/// - Encoding a sequence of (`SegmentLocalDocId`, `TermFreq`) pairs into a compressed byte stream.
/// - Decoding the byte stream back into the original pairs.
///
/// ## Encode input
///
/// The input to `encode()` is a slice of `(SegmentLocalDocId, TermFreq)` tuples.
/// Doc IDs must be doc-sorted (ascending order) for delta encoding to be effective.
///
/// ## Decode output
///
/// The decode API writes into **caller-provided output buffers** (`&mut Vec<SegmentLocalDocId>`
/// and `&mut Vec<TermFreq>`) rather than allocating owned decode structures. This design keeps
/// the codec layer scratch-ownership-agnostic; see DEC-12 and the TODO comment below.
///
/// The caller is responsible for clearing or reusing these buffers.
///
/// ## Codec marker
///
/// The encoded byte stream includes a 1-byte `CodecId` prefix.
/// The `encode()` method includes this prefix in the returned bytes.
/// The `decode()` method expects the bytes to start with the marker.
pub trait Codec {
    /// Return the codec ID for this implementation.
    fn id(&self) -> CodecId;

    /// Encode a sequence of (`SegmentLocalDocId`, `TermFreq`) tuples into a compressed byte stream.
    ///
    /// The returned bytes include a 1-byte codec ID prefix.
    ///
    /// The v1 codec layer encodes only `(SegmentLocalDocId, TermFreq)`. Posting **positions**
    /// (`Posting::positions`) are intentionally out of scope here — positions/TF
    /// layering is decided in ITER-0003 (STORY-0080); a future codec or a parallel
    /// positions section will carry them without changing this format.
    ///
    /// # Arguments
    ///
    /// - `postings`: slice of (`SegmentLocalDocId`, `TermFreq`) pairs, must be doc-sorted (ascending).
    ///
    /// # Panics
    ///
    /// May panic if postings are not doc-sorted.
    fn encode(&self, postings: &[(SegmentLocalDocId, TermFreq)]) -> Vec<u8>;

    /// Decode a byte stream into doc IDs and term frequencies.
    ///
    /// This method writes decoded values into caller-provided output buffers.
    /// The input bytes must start with a valid codec ID marker.
    ///
    /// The method validates the codec ID and delegates to the appropriate codec
    /// decoder, which writes exactly `len(postings)` values into `out_docs` and
    /// `out_tfs` in doc-ascending order.
    ///
    /// # Arguments
    ///
    /// - `bytes`: compressed postings with a 1-byte codec ID prefix.
    /// - `out_docs`: output buffer for decoded doc IDs (will be cleared and refilled).
    /// - `out_tfs`: output buffer for decoded term frequencies (will be cleared and refilled).
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or a `CodecError` if decoding fails.
    ///
    /// # TODO(ITER-0003): `DecodeScratch` wrapper
    ///
    /// The decode-scratch ownership model (e.g., a `DecodeScratch` struct holding
    /// these buffers and reusable across multiple cursors) is designed and built
    /// in ITER-0003 (STORY-0079). This codec layer intentionally does NOT define
    /// a decode-scratch type so that decision is not boxed in here.
    fn decode(
        &self,
        bytes: &[u8],
        out_docs: &mut Vec<SegmentLocalDocId>,
        out_tfs: &mut Vec<TermFreq>,
    ) -> Result<(), CodecError>;
}

/// Helper to encode a u32 as LEB128 varint.
///
/// Writes 1–5 bytes into `out` (a `u32` LEB128 is at most 5 bytes, so a fixed
/// `[u8; 5]` buffer can never overflow), returns the number of bytes written.
fn encode_varint(value: u32, out: &mut [u8; 5]) -> usize {
    let mut v = value;
    let mut len = 0;
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out[len] = byte;
        len += 1;
        if v == 0 {
            break;
        }
    }
    len
}

/// Helper to decode a u32 from LEB128 varint.
///
/// Returns `(value, bytes_consumed)` on success, or `CodecError::InvalidVarint` on failure.
fn decode_varint(bytes: &[u8]) -> Result<(u32, usize), CodecError> {
    let mut value = 0_u32;
    let mut shift = 0;
    let mut pos = 0;

    loop {
        if pos >= bytes.len() {
            return Err(CodecError::Truncated);
        }

        let byte = bytes[pos];
        pos += 1;

        value |= ((byte & 0x7f) as u32) << shift;

        if (byte & 0x80) == 0 {
            return Ok((value, pos));
        }

        shift += 7;
        if shift >= 32 {
            return Err(CodecError::InvalidVarint);
        }
    }
}

/// Delta-encoding codec: single-block varint encoding.
///
/// This codec encodes the postings list as a single block with:
/// - Delta-encoded doc IDs (deltas from previous doc, first delta from 0).
/// - Varint-encoded term frequencies (parallel to doc stream).
///
/// This is a simpler, single-block alternative to `BlockDelta` for small postings lists.
#[derive(Clone, Copy, Debug)]
pub struct DeltaVarintCodec;

impl Codec for DeltaVarintCodec {
    fn id(&self) -> CodecId {
        CodecId::DeltaVarint
    }

    fn encode(&self, postings: &[(SegmentLocalDocId, TermFreq)]) -> Vec<u8> {
        let mut result = Vec::with_capacity(postings.len() * 5 + 10);
        result.push(CodecId::DeltaVarint.to_u8());

        let mut buf = [0_u8; 5];
        let mut prev_doc = 0_u32;

        for (doc_id, tf) in postings {
            // Postings MUST be doc-sorted ascending (the documented precondition).
            // `checked_sub` turns a violation into a deterministic panic rather than
            // a silently-wrapping delta that would corrupt the encoded stream.
            let doc_id_u32 = doc_id.get();
            let delta = doc_id_u32
                .checked_sub(prev_doc)
                .expect("postings must be doc-sorted ascending");
            let bytes_written = encode_varint(delta, &mut buf);
            result.extend_from_slice(&buf[..bytes_written]);

            let bytes_written = encode_varint(tf.get(), &mut buf);
            result.extend_from_slice(&buf[..bytes_written]);

            prev_doc = doc_id_u32;
        }

        result
    }

    fn decode(
        &self,
        bytes: &[u8],
        out_docs: &mut Vec<SegmentLocalDocId>,
        out_tfs: &mut Vec<TermFreq>,
    ) -> Result<(), CodecError> {
        out_docs.clear();
        out_tfs.clear();

        if bytes.is_empty() {
            return Err(CodecError::Truncated);
        }

        let marker = bytes[0];
        if marker != CodecId::DeltaVarint.to_u8() {
            return Err(CodecError::BadMarker(marker));
        }

        let mut pos = 1;
        let mut prev_doc = 0_u32;

        while pos < bytes.len() {
            let (delta, delta_len) = decode_varint(&bytes[pos..])?;
            pos += delta_len;

            if pos >= bytes.len() {
                return Err(CodecError::Truncated);
            }

            let doc_id = prev_doc
                .checked_add(delta)
                .ok_or(CodecError::InvalidVarint)?;
            let (tf, tf_len) = decode_varint(&bytes[pos..])?;
            pos += tf_len;

            out_docs.push(SegmentLocalDocId::new(doc_id));
            out_tfs.push(TermFreq::new(tf));
            prev_doc = doc_id;
        }

        Ok(())
    }
}

/// Block-delta codec: multi-block encoding with 128-doc blocks.
///
/// This codec divides postings into fixed-size blocks of `BLOCK_DOC_COUNT` documents.
/// Each block is independently decodable, enabling selective decode and block-aware
/// traversal for future pruning (Phase 3).
///
/// ## Block format
///
/// The block format (not Rust code):
///
/// ```text
/// block := block_header doc_id_stream tf_stream
/// block_header := varint(doc_count) varint(first_doc) varint(last_doc) varint(doc_bytes_len)
/// doc_id_stream := varint(first_delta) varint(delta)*      # deltas from previous doc
/// tf_stream     := varint(tf)*                              # one per doc, parallel to doc stream
/// ```
///
/// The `doc_count` in the header is used by readers but is implicit from input length.
/// The `first_doc` is stored as an absolute value (delta from 0), so blocks are self-contained.
/// The `doc_bytes_len` allows readers to skip to the TF stream without decoding doc deltas.
#[derive(Clone, Copy, Debug)]
pub struct BlockDeltaCodec;

impl Codec for BlockDeltaCodec {
    fn id(&self) -> CodecId {
        CodecId::BlockDelta
    }

    fn encode(&self, postings: &[(SegmentLocalDocId, TermFreq)]) -> Vec<u8> {
        let mut result = Vec::with_capacity(postings.len() * 5 + 100);
        result.push(CodecId::BlockDelta.to_u8());

        let mut buf = [0_u8; 5];

        for block_chunk in postings.chunks(BLOCK_DOC_COUNT) {
            let doc_count = block_chunk.len();
            let first_doc = block_chunk[0].0.get();
            let last_doc = block_chunk[block_chunk.len() - 1].0.get();

            // Encode block header: doc_count, first_doc, last_doc, doc_bytes_len.
            // We'll first encode doc stream in a temporary buffer to get its length.

            // SAFETY: doc_count is usize from chunk size, bounded by BLOCK_DOC_COUNT (128).
            #[expect(
                clippy::cast_possible_truncation,
                reason = "doc_count bounded by BLOCK_DOC_COUNT"
            )]
            let bytes = encode_varint(doc_count as u32, &mut buf);
            result.extend_from_slice(&buf[..bytes]);

            let bytes = encode_varint(first_doc, &mut buf);
            result.extend_from_slice(&buf[..bytes]);

            let bytes = encode_varint(last_doc, &mut buf);
            result.extend_from_slice(&buf[..bytes]);

            // Encode doc ID stream in a temporary buffer to determine its length.
            let mut doc_stream = Vec::new();
            let mut prev_doc = 0_u32;
            for (doc_id, _) in block_chunk {
                // Doc-sorted precondition (see DeltaVarintCodec::encode): a violation
                // panics deterministically rather than wrapping into a corrupt delta.
                let doc_id_u32 = doc_id.get();
                let delta = doc_id_u32
                    .checked_sub(prev_doc)
                    .expect("postings must be doc-sorted ascending");
                let bytes = encode_varint(delta, &mut buf);
                doc_stream.extend_from_slice(&buf[..bytes]);
                prev_doc = doc_id_u32;
            }

            // Now encode doc_bytes_len.
            // SAFETY: doc_bytes_len is per-block, bounded by BLOCK_DOC_COUNT * varint_max_bytes.
            let doc_bytes_len = doc_stream.len();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "doc_bytes_len bounded by BLOCK_DOC_COUNT * 5"
            )]
            let bytes = encode_varint(doc_bytes_len as u32, &mut buf);
            result.extend_from_slice(&buf[..bytes]);

            // Append the doc stream.
            result.extend_from_slice(&doc_stream);

            // Encode TF values.
            for (_, tf) in block_chunk {
                let bytes = encode_varint(tf.get(), &mut buf);
                result.extend_from_slice(&buf[..bytes]);
            }
        }

        result
    }

    fn decode(
        &self,
        bytes: &[u8],
        out_docs: &mut Vec<SegmentLocalDocId>,
        out_tfs: &mut Vec<TermFreq>,
    ) -> Result<(), CodecError> {
        out_docs.clear();
        out_tfs.clear();

        if bytes.is_empty() {
            return Err(CodecError::Truncated);
        }

        let marker = bytes[0];
        if marker != CodecId::BlockDelta.to_u8() {
            return Err(CodecError::BadMarker(marker));
        }

        let mut pos = 1;

        while pos < bytes.len() {
            // Decode block header.
            let (doc_count, bytes_read) = decode_varint(&bytes[pos..])?;
            pos += bytes_read;
            let doc_count = doc_count as usize;

            if doc_count == 0 {
                return Err(CodecError::InvalidBlockCount);
            }

            let (first_doc, bytes_read) = decode_varint(&bytes[pos..])?;
            pos += bytes_read;

            let (last_doc, bytes_read) = decode_varint(&bytes[pos..])?;
            pos += bytes_read;

            let (doc_bytes_len, bytes_read) = decode_varint(&bytes[pos..])?;
            pos += bytes_read;
            let doc_bytes_len = doc_bytes_len as usize;

            if pos + doc_bytes_len > bytes.len() {
                return Err(CodecError::Truncated);
            }

            // Decode doc ID stream.
            let doc_stream_start = pos;
            let doc_stream_end = pos + doc_bytes_len;
            let doc_stream = &bytes[doc_stream_start..doc_stream_end];

            let mut doc_pos = 0;
            let mut prev_doc = 0_u32;
            let block_doc_start = out_docs.len();

            for _ in 0..doc_count {
                if doc_pos >= doc_stream.len() {
                    return Err(CodecError::Truncated);
                }
                let (delta, bytes_read) = decode_varint(&doc_stream[doc_pos..])?;
                doc_pos += bytes_read;

                let doc_id = prev_doc
                    .checked_add(delta)
                    .ok_or(CodecError::InvalidVarint)?;
                out_docs.push(SegmentLocalDocId::new(doc_id));
                prev_doc = doc_id;
            }

            // Validate the block header's doc-range against the decoded stream.
            // This makes `first_doc`/`last_doc` (carried for ITER-0003 block-skip and
            // ITER-0005 WAND doc-range) self-checking rather than dead bytes, and
            // detects corruption. `prev_doc` now holds the block's last decoded doc.
            if out_docs[block_doc_start].get() != first_doc || prev_doc != last_doc {
                return Err(CodecError::BlockHeaderMismatch);
            }

            pos = doc_stream_end;

            // Decode TF stream (parallel to doc stream).
            for _ in 0..doc_count {
                if pos >= bytes.len() {
                    return Err(CodecError::Truncated);
                }
                let (tf, bytes_read) = decode_varint(&bytes[pos..])?;
                pos += bytes_read;
                out_tfs.push(TermFreq::new(tf));
            }
        }

        Ok(())
    }
}

/// Decode a postings byte stream using any codec.
///
/// This free function dispatches on the codec ID marker and calls the appropriate
/// codec's decode method.
///
/// # Arguments
///
/// - `bytes`: the encoded postings (must start with a valid `CodecId` marker).
/// - `out_docs`: output buffer for doc IDs.
/// - `out_tfs`: output buffer for term frequencies.
///
/// # Returns
///
/// `Ok(())` on success, or a `CodecError` on failure.
pub fn decode_any(
    bytes: &[u8],
    out_docs: &mut Vec<SegmentLocalDocId>,
    out_tfs: &mut Vec<TermFreq>,
) -> Result<(), CodecError> {
    if bytes.is_empty() {
        return Err(CodecError::Truncated);
    }

    let marker = bytes[0];
    match CodecId::from_u8(marker) {
        Some(CodecId::DeltaVarint) => DeltaVarintCodec.decode(bytes, out_docs, out_tfs),
        Some(CodecId::BlockDelta) => BlockDeltaCodec.decode(bytes, out_docs, out_tfs),
        None => Err(CodecError::BadMarker(marker)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    // ===== Varint Tests =====

    #[test]
    fn test_varint_encode_decode_zero() {
        let mut buf = [0_u8; 5];
        let len = encode_varint(0, &mut buf);
        assert_eq!(len, 1);
        assert_eq!(buf[0], 0);

        let (decoded, bytes_read) = decode_varint(&buf[..len]).unwrap();
        assert_eq!(decoded, 0);
        assert_eq!(bytes_read, 1);
    }

    #[test]
    fn test_varint_encode_decode_small() {
        let mut buf = [0_u8; 5];
        let len = encode_varint(42, &mut buf);
        assert_eq!(len, 1);

        let (decoded, bytes_read) = decode_varint(&buf[..len]).unwrap();
        assert_eq!(decoded, 42);
        assert_eq!(bytes_read, 1);
    }

    #[test]
    fn test_varint_encode_decode_large() {
        let mut buf = [0_u8; 5];
        let len = encode_varint(16384, &mut buf);
        assert_eq!(len, 3);

        let (decoded, bytes_read) = decode_varint(&buf[..len]).unwrap();
        assert_eq!(decoded, 16384);
        assert_eq!(bytes_read, 3);
    }

    #[test]
    fn test_varint_encode_decode_max() {
        let mut buf = [0_u8; 5];
        let len = encode_varint(u32::MAX, &mut buf);
        assert_eq!(len, 5);

        let (decoded, bytes_read) = decode_varint(&buf[..len]).unwrap();
        assert_eq!(decoded, u32::MAX);
        assert_eq!(bytes_read, 5);
    }

    #[test]
    fn test_varint_decode_truncated() {
        let bytes = [0x80]; // Incomplete varint.
        let result = decode_varint(&bytes);
        assert_eq!(result, Err(CodecError::Truncated));
    }

    // ===== DeltaVarint Codec Tests =====

    #[test]
    fn test_delta_varint_round_trip_empty() {
        let codec = DeltaVarintCodec;
        let postings: &[(SegmentLocalDocId, TermFreq)] = &[];

        let encoded = codec.encode(postings);
        assert_eq!(encoded.len(), 1); // Only the marker byte.

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(docs.len(), 0);
        assert_eq!(tfs.len(), 0);
    }

    #[test]
    fn test_delta_varint_round_trip_single() {
        let codec = DeltaVarintCodec;
        let postings = [(SegmentLocalDocId::new(100), TermFreq::new(5))];

        let encoded = codec.encode(&postings);

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(docs, vec![SegmentLocalDocId::new(100)]);
        assert_eq!(tfs, vec![TermFreq::new(5)]);
    }

    #[test]
    fn test_delta_varint_round_trip_multiple() {
        let codec = DeltaVarintCodec;
        let postings = [
            (SegmentLocalDocId::new(10), TermFreq::new(1)),
            (SegmentLocalDocId::new(20), TermFreq::new(2)),
            (SegmentLocalDocId::new(35), TermFreq::new(3)),
            (SegmentLocalDocId::new(100), TermFreq::new(10)),
        ];

        let encoded = codec.encode(&postings);

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(
            docs,
            vec![
                SegmentLocalDocId::new(10),
                SegmentLocalDocId::new(20),
                SegmentLocalDocId::new(35),
                SegmentLocalDocId::new(100),
            ]
        );
        assert_eq!(
            tfs,
            vec![
                TermFreq::new(1),
                TermFreq::new(2),
                TermFreq::new(3),
                TermFreq::new(10),
            ]
        );
    }

    #[test]
    fn test_delta_varint_compression() {
        let codec = DeltaVarintCodec;
        let postings = [
            (SegmentLocalDocId::new(100), TermFreq::new(5)),
            (SegmentLocalDocId::new(105), TermFreq::new(3)),
            (SegmentLocalDocId::new(110), TermFreq::new(7)),
            (SegmentLocalDocId::new(200), TermFreq::new(2)),
        ];

        let encoded = codec.encode(&postings);
        let uncompressed = postings.len() * 8; // 4 bytes doc + 4 bytes tf.

        // Encoded should be smaller than uncompressed.
        // With deltas and varints, this should be significantly smaller.
        assert!(
            encoded.len() < uncompressed,
            "encoded {} >= uncompressed {}",
            encoded.len(),
            uncompressed
        );
    }

    #[test]
    fn test_delta_varint_bad_marker() {
        let codec = DeltaVarintCodec;
        let bad_bytes = [5_u8, 100, 5]; // Bad marker.

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        let result = codec.decode(&bad_bytes, &mut docs, &mut tfs);

        assert_eq!(result, Err(CodecError::BadMarker(5)));
    }

    // ===== BlockDelta Codec Tests =====

    #[test]
    fn test_block_delta_round_trip_empty() {
        let codec = BlockDeltaCodec;
        let postings: &[(SegmentLocalDocId, TermFreq)] = &[];

        let encoded = codec.encode(postings);
        assert_eq!(encoded.len(), 1); // Only marker.

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(docs.len(), 0);
        assert_eq!(tfs.len(), 0);
    }

    #[test]
    fn test_block_delta_round_trip_single_block() {
        let codec = BlockDeltaCodec;
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..50)
            .map(|i| (SegmentLocalDocId::new(i * 10), TermFreq::new(i % 10 + 1)))
            .collect();

        let encoded = codec.encode(&postings);

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        let expected_docs: Vec<SegmentLocalDocId> = postings.iter().map(|(d, _)| *d).collect();
        let expected_tfs: Vec<TermFreq> = postings.iter().map(|(_, t)| *t).collect();

        assert_eq!(docs, expected_docs);
        assert_eq!(tfs, expected_tfs);
    }

    #[test]
    fn test_block_delta_round_trip_multiple_blocks() {
        let codec = BlockDeltaCodec;
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..300)
            .map(|i| (SegmentLocalDocId::new(i * 5), TermFreq::new(i % 7 + 1)))
            .collect();

        let encoded = codec.encode(&postings);

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        let expected_docs: Vec<SegmentLocalDocId> = postings.iter().map(|(d, _)| *d).collect();
        let expected_tfs: Vec<TermFreq> = postings.iter().map(|(_, t)| *t).collect();

        assert_eq!(docs, expected_docs);
        assert_eq!(tfs, expected_tfs);
    }

    #[test]
    fn test_block_delta_compression() {
        let codec = BlockDeltaCodec;
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..200)
            .map(|i| (SegmentLocalDocId::new(i * 3), TermFreq::new(i % 5 + 1)))
            .collect();

        let encoded = codec.encode(&postings);
        let uncompressed = postings.len() * 8;

        assert!(
            encoded.len() < uncompressed,
            "encoded {} >= uncompressed {}",
            encoded.len(),
            uncompressed
        );
    }

    #[test]
    fn test_block_delta_independent_decode() {
        let codec = BlockDeltaCodec;
        // Create a large postings list (> BLOCK_DOC_COUNT).
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..300)
            .map(|i| (SegmentLocalDocId::new(i * 2), TermFreq::new(i % 3 + 1)))
            .collect();

        let encoded = codec.encode(&postings);

        // Verify full decode.
        let mut all_docs = Vec::new();
        let mut all_tfs = Vec::new();
        codec.decode(&encoded, &mut all_docs, &mut all_tfs).unwrap();

        assert_eq!(all_docs.len(), 300);
        assert_eq!(all_tfs.len(), 300);

        // Verify that blocks are self-contained by decoding and checking structure.
        // (A full independent block decode would require a separate `decode_block` method,
        // which is deferred but noted in AC-2 proof; for now we verify the format is correct
        // by full decode and structural checks.)
        let expected_docs: Vec<SegmentLocalDocId> = postings.iter().map(|(d, _)| *d).collect();
        let expected_tfs: Vec<TermFreq> = postings.iter().map(|(_, t)| *t).collect();

        assert_eq!(all_docs, expected_docs);
        assert_eq!(all_tfs, expected_tfs);
    }

    #[test]
    fn test_block_delta_bad_marker() {
        let codec = BlockDeltaCodec;
        let bad_bytes = [99_u8]; // Bad marker.

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        let result = codec.decode(&bad_bytes, &mut docs, &mut tfs);

        assert_eq!(result, Err(CodecError::BadMarker(99)));
    }

    // ===== decode_any Tests =====

    #[test]
    fn test_decode_any_delta_varint() {
        let codec = DeltaVarintCodec;
        let postings = [
            (SegmentLocalDocId::new(10), TermFreq::new(1)),
            (SegmentLocalDocId::new(20), TermFreq::new(2)),
        ];
        let encoded = codec.encode(&postings);

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        decode_any(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(
            docs,
            vec![SegmentLocalDocId::new(10), SegmentLocalDocId::new(20)]
        );
        assert_eq!(tfs, vec![TermFreq::new(1), TermFreq::new(2)]);
    }

    #[test]
    fn test_decode_any_block_delta() {
        let codec = BlockDeltaCodec;
        let postings = [
            (SegmentLocalDocId::new(10), TermFreq::new(1)),
            (SegmentLocalDocId::new(20), TermFreq::new(2)),
        ];
        let encoded = codec.encode(&postings);

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        decode_any(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(
            docs,
            vec![SegmentLocalDocId::new(10), SegmentLocalDocId::new(20)]
        );
        assert_eq!(tfs, vec![TermFreq::new(1), TermFreq::new(2)]);
    }

    #[test]
    fn test_decode_any_bad_marker() {
        let bad_bytes = [99_u8];

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        let result = decode_any(&bad_bytes, &mut docs, &mut tfs);

        assert_eq!(result, Err(CodecError::BadMarker(99)));
    }

    // ===== Integration: Size Reduction Tests =====

    #[test]
    fn test_delta_varint_size_reduction_zipfian() {
        // Simulate a Zipfian-like distribution (long list with concentrated doc IDs).
        let codec = DeltaVarintCodec;
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..100)
            .map(|i| {
                let doc_id = (i * 50) as u32; // Sparse docs, large deltas within blocks.
                let tf = (1 + (i % 10)) as u32;
                (SegmentLocalDocId::new(doc_id), TermFreq::new(tf))
            })
            .collect();

        let encoded = codec.encode(&postings);
        let uncompressed_bytes = postings.len() * 8;

        assert!(encoded.len() < uncompressed_bytes);
    }

    #[test]
    fn test_block_delta_size_reduction_zipfian() {
        let codec = BlockDeltaCodec;
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..200)
            .map(|i| {
                let doc_id = (i * 50) as u32;
                let tf = (1 + (i % 10)) as u32;
                (SegmentLocalDocId::new(doc_id), TermFreq::new(tf))
            })
            .collect();

        let encoded = codec.encode(&postings);
        let uncompressed_bytes = postings.len() * 8;

        assert!(encoded.len() < uncompressed_bytes);
    }

    #[test]
    fn test_block_delta_exactly_128_docs() {
        let codec = BlockDeltaCodec;
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..128)
            .map(|i| {
                (
                    SegmentLocalDocId::new(i as u32),
                    TermFreq::new((i % 7 + 1) as u32),
                )
            })
            .collect();

        let encoded = codec.encode(&postings);
        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(docs.len(), 128, "Should have exactly 128 docs");
        let expected_docs: Vec<SegmentLocalDocId> =
            (0..128).map(|i| SegmentLocalDocId::new(i as u32)).collect();
        assert_eq!(docs, expected_docs);
    }

    #[test]
    fn test_block_delta_129_docs() {
        let codec = BlockDeltaCodec;
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..129)
            .map(|i| {
                (
                    SegmentLocalDocId::new(i as u32),
                    TermFreq::new((i % 7 + 1) as u32),
                )
            })
            .collect();

        let encoded = codec.encode(&postings);
        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(docs.len(), 129, "Should have 129 docs in 2 blocks");
        let expected_docs: Vec<SegmentLocalDocId> =
            (0..129).map(|i| SegmentLocalDocId::new(i as u32)).collect();
        assert_eq!(docs, expected_docs);
    }

    #[test]
    fn test_delta_varint_large_tf_values() {
        let codec = DeltaVarintCodec;
        // Test with TF values that require multi-byte varints (> 127)
        let postings = [
            (SegmentLocalDocId::new(10), TermFreq::new(200)),
            (SegmentLocalDocId::new(20), TermFreq::new(300)),
            (SegmentLocalDocId::new(35), TermFreq::new(16384)),
            (SegmentLocalDocId::new(100), TermFreq::new(32768)),
        ];

        let encoded = codec.encode(&postings);
        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(
            docs,
            vec![
                SegmentLocalDocId::new(10),
                SegmentLocalDocId::new(20),
                SegmentLocalDocId::new(35),
                SegmentLocalDocId::new(100),
            ]
        );
        assert_eq!(
            tfs,
            vec![
                TermFreq::new(200),
                TermFreq::new(300),
                TermFreq::new(16384),
                TermFreq::new(32768),
            ]
        );
    }

    #[test]
    fn test_block_delta_large_doc_gaps() {
        let codec = BlockDeltaCodec;
        // Simulate Zipfian: wide doc ID gaps within blocks
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..100)
            .map(|i| {
                (
                    SegmentLocalDocId::new((i as u32) * 10000),
                    TermFreq::new((i % 7 + 1) as u32),
                )
            })
            .collect();

        let encoded = codec.encode(&postings);
        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(docs.len(), 100);
        let expected_docs: Vec<SegmentLocalDocId> = (0..100)
            .map(|i| SegmentLocalDocId::new((i as u32) * 10000))
            .collect();
        assert_eq!(docs, expected_docs);
    }

    #[test]
    fn test_block_delta_block_independence_cross_boundary() {
        // Verify that block boundaries don't affect doc reconstruction.
        // This tests that the first_doc in each block is stored as an absolute value
        // (delta from 0), not as a delta from the previous block.
        let codec = BlockDeltaCodec;

        // Create a 256-doc list spanning 2 blocks.
        // Block 1: docs 0..127 (even IDs)
        // Block 2: docs 128..255 (even IDs)
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..256)
            .map(|i| (SegmentLocalDocId::new((i as u32) * 2), TermFreq::new(1)))
            .collect();

        let encoded = codec.encode(&postings);
        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        // Verify all docs were reconstructed correctly across block boundaries.
        assert_eq!(docs.len(), 256);
        for (i, &doc) in docs.iter().enumerate() {
            assert_eq!(
                doc.get(),
                u32::try_from(i).unwrap() * 2,
                "Doc at index {i} mismatch"
            );
        }
    }

    #[test]
    fn test_block_delta_non_unit_deltas() {
        // Test block boundaries with varying delta sizes.
        // This specifically checks that block-boundary docs have the right doc ID.
        let codec = BlockDeltaCodec;

        // Docs: 0, 100, 200, ..., 12700 (128 docs per block)
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = (0..256)
            .map(|i| {
                (
                    SegmentLocalDocId::new((i as u32) * 100),
                    TermFreq::new((i % 5 + 1) as u32),
                )
            })
            .collect();

        let encoded = codec.encode(&postings);
        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        codec.decode(&encoded, &mut docs, &mut tfs).unwrap();

        assert_eq!(docs.len(), 256);
        // Verify critical boundary points
        assert_eq!(docs[127].get(), 127 * 100, "Last doc of block 1");
        assert_eq!(
            docs[128].get(),
            128 * 100,
            "First doc of block 2 (critical boundary)"
        );
        assert_eq!(docs[255].get(), 255 * 100, "Last doc of block 2");
    }

    #[test]
    fn test_varint_over_long_six_bytes() {
        // Manually construct a 6-byte varint encoding (invalid).
        // After reading the 5th byte, shift=28. On the 6th byte, shift becomes 35 >= 32.
        let codec = DeltaVarintCodec;
        let mut bad_bytes = Vec::new();
        bad_bytes.push(CodecId::DeltaVarint.to_u8());
        // Encode a valid posting first (doc_id=10, tf=1)
        bad_bytes.extend_from_slice(&[10_u8, 1_u8]);
        // Append over-long varint: all bytes with MSB set
        bad_bytes.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01_u8]);

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        let result = codec.decode(&bad_bytes, &mut docs, &mut tfs);

        // Must reject over-long varint, not silently accept
        assert_eq!(result, Err(CodecError::InvalidVarint));
    }

    #[test]
    fn test_delta_varint_malformed_truncated_tf() {
        // Encode: doc_delta(10) + incomplete TF (0x80 without continuation).
        let codec = DeltaVarintCodec;
        let mut bad_bytes = Vec::new();
        bad_bytes.push(CodecId::DeltaVarint.to_u8());
        // First posting: varint(10) for delta, then 0x80 (MSB set, more bytes expected)
        bad_bytes.extend_from_slice(&[10_u8, 0x80_u8]);

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        let result = codec.decode(&bad_bytes, &mut docs, &mut tfs);

        // Should detect truncation, not crash
        assert_eq!(result, Err(CodecError::Truncated));
    }

    #[test]
    fn test_block_delta_doc_bytes_len_bounds_check() {
        // Create a block header claiming more doc bytes than available.
        let codec = BlockDeltaCodec;
        let mut bad_bytes = Vec::new();
        bad_bytes.push(CodecId::BlockDelta.to_u8());
        // Block header: doc_count=2, first=10, last=20, doc_bytes_len=100 (but only 2 bytes follow)
        bad_bytes.extend_from_slice(&[2_u8, 10_u8, 20_u8, 100_u8]);
        bad_bytes.extend_from_slice(&[10_u8, 1_u8]); // Only 2 bytes of doc stream

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        let result = codec.decode(&bad_bytes, &mut docs, &mut tfs);

        // Must reject, not read past end
        assert_eq!(result, Err(CodecError::Truncated));
    }

    #[test]
    #[should_panic(expected = "doc-sorted")]
    fn test_delta_varint_unsorted_input_panics() {
        // The doc-sorted precondition is enforced: an out-of-order doc id panics
        // deterministically instead of silently producing a corrupt (wrapped) delta.
        let codec = DeltaVarintCodec;
        let _ = codec.encode(&[
            (SegmentLocalDocId::new(10), TermFreq::new(1)),
            (SegmentLocalDocId::new(5), TermFreq::new(1)),
        ]);
    }

    #[test]
    #[should_panic(expected = "doc-sorted")]
    fn test_block_delta_unsorted_input_panics() {
        let codec = BlockDeltaCodec;
        let _ = codec.encode(&[
            (SegmentLocalDocId::new(10), TermFreq::new(1)),
            (SegmentLocalDocId::new(5), TermFreq::new(1)),
        ]);
    }

    #[test]
    fn test_block_delta_corrupt_header_doc_range_rejected() {
        // A block header whose first_doc/last_doc disagree with the doc stream is
        // rejected (the header range is validated, not ignored).
        let codec = BlockDeltaCodec;
        let encoded = codec.encode(&[
            (SegmentLocalDocId::new(3), TermFreq::new(1)),
            (SegmentLocalDocId::new(7), TermFreq::new(2)),
            (SegmentLocalDocId::new(11), TermFreq::new(3)),
        ]);

        // Header layout after the 1-byte marker: varint(doc_count) varint(first_doc) ...
        // doc_count=3 at index 1, first_doc=3 at index 2. Corrupt first_doc 3 -> 4.
        let mut corrupt = encoded.clone();
        assert_eq!(corrupt[2], 3, "expected first_doc varint at index 2");
        corrupt[2] = 4;

        let mut docs = Vec::new();
        let mut tfs = Vec::new();
        assert_eq!(
            codec.decode(&corrupt, &mut docs, &mut tfs),
            Err(CodecError::BlockHeaderMismatch)
        );
    }
}
