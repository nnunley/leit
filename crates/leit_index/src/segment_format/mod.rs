// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Segment format v1: fixed-layout, little-endian POD header with
//! u64 absolute offsets.

#[expect(
    unreachable_pub,
    reason = "header module items are re-exported from lib.rs"
)]
pub mod header;

#[expect(
    unreachable_pub,
    reason = "footer items are consumed by SegmentView; unreachable in the lib build until then"
)]
pub mod footer;

#[expect(
    unreachable_pub,
    reason = "reader items are consumed by SegmentView; unreachable in the lib build until then"
)]
pub mod readers;

#[expect(
    unreachable_pub,
    reason = "writer module is consumed by InMemoryIndex::to_segment_bytes; re-export pending until phase 3"
)]
pub mod writer;

#[expect(
    unreachable_pub,
    reason = "view module is exported via pub use SegmentView; unreachable_pub allows this pattern"
)]
pub mod view;

#[expect(
    unreachable_pub,
    reason = "block_meta module items are used internally for per-block metadata storage and will be re-exported once public block navigation is stabilized"
)]
pub mod block_meta;

pub use header::{FORMAT_VERSION, HEADER_SIZE, MAGIC, SegmentHeader};
pub use view::SegmentView;
