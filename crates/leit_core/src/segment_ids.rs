// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Segment-resident core ID types with a stable, zero-copy serialized representation.
//!
//! Unlike the in-memory index identifiers ([`FieldId`](crate::FieldId),
//! [`TermId`](crate::TermId), [`SegmentId`](crate::SegmentId)), the types in this
//! module are designed to appear **directly in mmap'd segment bytes**. Each is a
//! `#[repr(transparent)]` newtype over a 4-byte little-endian array, so a `&[u8]`
//! slice taken from a memory-mapped buffer can be viewed in place as `&[Id]` with
//! no allocation and no deserialization pass (see the Phase 2 architectural
//! decisions: zero-copy via `bytemuck`).
//!
//! The inner representation is `[u8; 4]` holding the **little-endian** bytes of a
//! `u32`, so the on-disk form is identical on every host (portable across
//! endianness). Because the storage is a byte array, ordering is implemented
//! by numeric value rather than raw byte order.

use bytemuck::{Pod, Zeroable};

/// Define a segment-resident ID newtype over a little-endian 4-byte value.
macro_rules! segment_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Fixed-width 4-byte little-endian value; viewable in place from mmap'd
        /// segment bytes via `bytemuck` (`Pod`).
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
        #[repr(transparent)]
        pub struct $name([u8; 4]);

        impl $name {
            #[doc = concat!("Create a new `", stringify!($name), "` from a raw `u32`.")]
            #[must_use]
            pub const fn new(value: u32) -> Self {
                Self(value.to_le_bytes())
            }

            #[doc = concat!("Get the raw `u32` value of this `", stringify!($name), "`.")]
            #[must_use]
            pub const fn get(self) -> u32 {
                u32::from_le_bytes(self.0)
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.get())
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.get().cmp(&other.get())
            }
        }

        impl From<u32> for $name {
            fn from(value: u32) -> Self {
                Self::new(value)
            }
        }

        impl From<$name> for u32 {
            fn from(value: $name) -> Self {
                value.get()
            }
        }
    };
}

segment_id!(BlockId, "Identifier of a postings block within a segment.");
segment_id!(FilterExprId, "Identifier of a stored filter expression.");
segment_id!(
    SegmentOrd,
    "Ordinal position of a segment within a multi-segment index."
);
segment_id!(
    SegmentLocalDocId,
    "Document identifier local to a single segment (segment-relative doc ID)."
);

#[cfg(test)]
mod tests {
    use super::*;

    /// SCENARIO-0005: ID type serialization round-trip — each type serializes to
    /// a fixed-width 4-byte little-endian value and round-trips losslessly as both
    /// a single value and a zero-copy slice view.
    #[test]
    fn test_single_value_little_endian_roundtrip() {
        let id = BlockId::new(0xDEAD_BEEF);

        // Serialized form is exactly the 4-byte little-endian encoding.
        let bytes = bytemuck::bytes_of(&id);
        assert_eq!(bytes, &0xDEAD_BEEF_u32.to_le_bytes());
        assert_eq!(bytes.len(), 4);

        // Zero-copy view back to the typed value.
        let back: &BlockId = bytemuck::from_bytes(bytes);
        assert_eq!(*back, id);
        assert_eq!(back.get(), 0xDEAD_BEEF);
    }

    #[test]
    fn test_slice_zero_copy_view_roundtrip() {
        let ids = [
            SegmentLocalDocId::new(1),
            SegmentLocalDocId::new(0),
            SegmentLocalDocId::new(u32::MAX),
            SegmentLocalDocId::new(42),
        ];

        // The array's bytes are the concatenated little-endian values.
        let bytes: &[u8] = bytemuck::cast_slice(&ids);
        assert_eq!(bytes.len(), 4 * ids.len());
        assert_eq!(&bytes[0..4], &1_u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &42_u32.to_le_bytes());

        // A &[u8] is a zero-copy view of &[SegmentLocalDocId].
        let view: &[SegmentLocalDocId] = bytemuck::cast_slice(bytes);
        assert_eq!(view, ids.as_slice());
    }

    #[test]
    fn test_validated_reads_use_try_cast_variants() {
        // SCENARIO-0005 (AC-2 validated-read obligation): reads from untrusted
        // segment bytes go through the fallible `try_*` casts, which return `Err`
        // on a malformed slice instead of panicking.

        // Correctly-sized 4-byte slice -> Ok.
        let raw = 0x1234_5678_u32.to_le_bytes();
        let ok: &BlockId = bytemuck::try_from_bytes(&raw).expect("4-byte slice is a valid BlockId");
        assert_eq!(ok.get(), 0x1234_5678);

        // Wrong-length slice -> Err, never a panic.
        let too_short = [0_u8; 3];
        assert!(bytemuck::try_from_bytes::<BlockId>(&too_short).is_err());

        // try_cast_slice yields a zero-copy &[Id] view for an exact multiple...
        let ids = [SegmentLocalDocId::new(5), SegmentLocalDocId::new(6)];
        let bytes: &[u8] = bytemuck::cast_slice(&ids);
        let view: &[SegmentLocalDocId] =
            bytemuck::try_cast_slice(bytes).expect("8 bytes round-trips to 2 ids");
        assert_eq!(view, ids.as_slice());

        // ...and rejects a length that is not a whole number of ids.
        let ragged = [0_u8; 7];
        assert!(bytemuck::try_cast_slice::<u8, SegmentLocalDocId>(&ragged).is_err());
    }

    #[test]
    fn test_unaligned_view_from_offset() {
        // [u8; 4] storage is alignment-1, so views work from any byte offset
        // (mmap safety) — bytemuck's cast succeeds regardless of source alignment.
        let mut buf = [0_u8; 5];
        buf[1..5].copy_from_slice(&7_u32.to_le_bytes());
        let id: &SegmentOrd = bytemuck::from_bytes(&buf[1..5]);
        assert_eq!(id.get(), 7);
    }

    #[test]
    fn test_all_id_types_roundtrip_and_convert() {
        assert_eq!(BlockId::new(10).get(), 10);
        assert_eq!(FilterExprId::new(20).get(), 20);
        assert_eq!(SegmentOrd::new(30).get(), 30);
        assert_eq!(SegmentLocalDocId::new(40).get(), 40);

        // u32 <-> ID conversions.
        assert_eq!(u32::from(FilterExprId::from(99_u32)), 99);
    }

    #[test]
    fn test_ordering_is_numeric_not_byte_order() {
        // Little-endian byte storage must not corrupt numeric ordering.
        assert!(BlockId::new(2) < BlockId::new(256));
        assert!(SegmentLocalDocId::new(0x00FF_0000) < SegmentLocalDocId::new(0x0100_0000));
    }

    #[test]
    fn test_debug_shows_numeric_value() {
        extern crate alloc;
        assert_eq!(alloc::format!("{:?}", BlockId::new(7)), "BlockId(7)");
    }
}
