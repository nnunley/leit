// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Phase 2 segment format v1 (DEC-05): fixed-layout, little-endian POD header with
//! u64 absolute offsets.

#[expect(
    unreachable_pub,
    reason = "header module items are re-exported from lib.rs"
)]
pub mod header;

#[expect(
    unreachable_pub,
    reason = "footer items are consumed by SegmentView (pub + re-exported in T7); unreachable in the lib build until then"
)]
pub mod footer;

#[expect(
    unreachable_pub,
    reason = "reader items are consumed by SegmentView (pub + re-exported in T7); unreachable in the lib build until then"
)]
pub mod readers;

#[expect(
    unreachable_pub,
    reason = "writer module is consumed by InMemoryIndex::to_segment_bytes; re-export pending until phase 3"
)]
pub mod writer;

#[expect(
    unreachable_pub,
    reason = "SegmentView is pub(crate) until T7 promotes it to pub and flips the lib.rs re-export"
)]
pub mod view;

pub use header::{FORMAT_VERSION, HEADER_SIZE, MAGIC, SegmentHeader};
