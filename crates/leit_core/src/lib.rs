// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Core types and traits for Leit retrieval system.
//!
//! This crate provides foundational types used throughout the Leit ecosystem:
//! - Typed identifiers for fields, terms, segments, and query nodes
//! - Entity ID abstraction for application-defined identifiers
//! - Score type for retrieval scoring
//! - Hit type for search results
//! - Error types for core operations
//! - Scratch space and workspace traits for memory management

#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::fmt;
use core::hash::Hash;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

pub mod segment_ids;
pub use segment_ids::{BlockId, FilterExprId, SegmentLocalDocId, SegmentOrd, TermFreq};

/// Unique identifier for a field in an index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct FieldId(pub u32);

impl FieldId {
    /// Create a new field ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Unique identifier for a term in the dictionary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TermId(pub u32);

impl TermId {
    /// Create a new term ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Unique identifier for a segment in an index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SegmentId(pub u32);

impl SegmentId {
    /// Create a new segment ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Unique identifier for a node in a query program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct QueryNodeId(pub u32);

impl QueryNodeId {
    /// Create a new query node ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Identifier for a cursor slot during query execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CursorSlotId(pub u32);

impl CursorSlotId {
    /// Create a new cursor slot ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Identifier for a filter slot during query execution.
///
/// Indexes into an application-provided [`FilterEvaluator`] at execution time.
/// Unlike positional IDs, a default slot ID of 0 has no meaningful semantics,
/// so this type does not derive `Default`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct FilterSlotId(u32);

impl FilterSlotId {
    /// Create a new filter slot ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Trait for entity identifiers.
///
/// This trait is intentionally minimal - it does NOT require `Send + Sync`
/// to maintain `no_std` compatibility in kernel crates. Threading bounds
/// should be added at higher layers when needed.
pub trait EntityId: Copy + Eq + Hash + fmt::Debug + Ord {}

impl EntityId for u32 {}
impl EntityId for u64 {}
impl EntityId for i32 {}
impl EntityId for i64 {}

/// Evaluates application-provided filter predicates during query execution.
///
/// Implementations dispatch on [`FilterSlotId`] to evaluate arbitrary predicates
/// against candidate entities. The `id` is the same entity ID the caller
/// supplied during index construction.
pub trait FilterEvaluator<Id: EntityId> {
    /// Evaluate whether the entity with the given ID passes the filter.
    fn evaluate(&self, slot: FilterSlotId, id: &Id) -> bool;

    /// Return the filter slots this evaluator handles.
    ///
    /// These slots are used during planning to wrap the query with
    /// `ExternalFilter` nodes. An empty slice means no filtering.
    fn slots(&self) -> &[FilterSlotId];
}

/// No-op filter evaluator. All candidates pass.
///
/// When used as a type parameter, the compiler inlines the constant `true`
/// return and eliminates all filter checks via monomorphization.
#[derive(Clone, Copy, Debug)]
pub struct NoFilter;

impl<Id: EntityId> FilterEvaluator<Id> for NoFilter {
    fn evaluate(&self, _slot: FilterSlotId, _id: &Id) -> bool {
        true
    }

    fn slots(&self) -> &[FilterSlotId] {
        &[]
    }
}

/// A retrieval score.
///
/// This is a newtype over `f32` to provide type safety around finite scoring values.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Score(f32);

// SAFETY: Score guarantees finiteness (no NaN), so PartialEq is reflexive.
impl Eq for Score {}

impl Score {
    /// Score of zero.
    pub const ZERO: Self = Self(0.0);

    /// Score of one (perfect match baseline).
    pub const ONE: Self = Self(1.0);

    /// Minimum representable finite score.
    pub const MIN: Self = Self(f32::MIN);

    /// Maximum representable finite score.
    pub const MAX: Self = Self(f32::MAX);

    /// Create a new score from any finite `f32`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is NaN or infinite.
    pub const fn new(value: f32) -> Self {
        assert!(value.is_finite(), "score must be finite");
        Self(value)
    }

    /// Get the raw f32 value.
    pub const fn as_f32(self) -> f32 {
        self.0
    }

    /// Converts an arithmetic result to a `Score`, clamping non-finite values.
    ///
    /// - `NaN` maps to [`Score::ZERO`]
    /// - `+Inf` maps to [`Score::MAX`]
    /// - `-Inf` maps to [`Score::MIN`]
    /// - Finite values are clamped to `[f32::MIN, f32::MAX]`
    pub fn from_arithmetic_result(value: f32) -> Self {
        if value.is_nan() {
            Self::ZERO
        } else if value == f32::INFINITY {
            Self::MAX
        } else if value == f32::NEG_INFINITY {
            Self::MIN
        } else {
            Self(value.clamp(f32::MIN, f32::MAX))
        }
    }
}

/// Error returned when converting a non-finite `f32` to a [`Score`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonFiniteScoreError;

impl fmt::Display for NonFiniteScoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "score must be finite (not NaN or infinite)")
    }
}

impl core::error::Error for NonFiniteScoreError {}

impl TryFrom<f32> for Score {
    type Error = NonFiniteScoreError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(NonFiniteScoreError)
        }
    }
}

impl From<Score> for f32 {
    fn from(score: Score) -> Self {
        score.0
    }
}

impl Add for Score {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_arithmetic_result(self.0 + rhs.0)
    }
}

impl AddAssign for Score {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Score {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_arithmetic_result(self.0 - rhs.0)
    }
}

impl SubAssign for Score {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Score {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::from_arithmetic_result(self.0 * rhs)
    }
}

impl MulAssign<f32> for Score {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

#[cfg(feature = "std")]
impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

/// A scored search result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredHit<Id: EntityId> {
    /// The entity identifier.
    pub id: Id,
    /// The retrieval score.
    pub score: Score,
}

impl<Id: EntityId> ScoredHit<Id> {
    /// Create a new hit.
    pub const fn new(id: Id, score: Score) -> Self {
        Self { id, score }
    }

    /// Create a hit with a perfect score (1.0).
    pub const fn perfect(id: Id) -> Self {
        Self::new(id, Score::ONE)
    }

    /// Create a hit with a zero score.
    pub const fn zero(id: Id) -> Self {
        Self::new(id, Score::ZERO)
    }

    /// Check if this hit has a zero score.
    pub fn is_zero(&self) -> bool {
        self.score == Score::ZERO
    }
}

impl<Id: EntityId> Eq for ScoredHit<Id> {}

impl<Id: EntityId> PartialOrd for ScoredHit<Id> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Id: EntityId> Ord for ScoredHit<Id> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Higher scores compare as greater, then IDs provide a stable tiebreaker.
        match self.score.partial_cmp(&other.score) {
            Some(core::cmp::Ordering::Equal) | None => self.id.cmp(&other.id),
            Some(ord) => ord,
        }
    }
}

#[cfg(feature = "std")]
impl<Id: EntityId> fmt::Display for ScoredHit<Id> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScoredHit({:?}, {:.4})", self.id, self.score.as_f32())
    }
}

/// Core error types for Leit operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoreError {
    /// An invalid field ID was encountered.
    InvalidFieldId(u32),
    /// An invalid term ID was encountered.
    InvalidTermId(u32),
    /// A buffer was too small for the operation.
    BufferTooSmall {
        /// Required size.
        required: u32,
        /// Actual size.
        actual: u32,
    },
}

impl core::error::Error for CoreError {}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFieldId(id) => write!(f, "invalid field ID: {id}"),
            Self::InvalidTermId(id) => write!(f, "invalid term ID: {id}"),
            Self::BufferTooSmall { required, actual } => {
                write!(f, "buffer too small: required {required}, got {actual}")
            }
        }
    }
}

/// Trait for reusable scratch memory.
pub trait ScratchSpace {
    /// Clear the scratch space for reuse.
    fn clear(&mut self);
}

/// Trait for execution workspace memory.
///
/// Workspaces extend scratch spaces with additional capabilities
/// for query execution.
pub trait Workspace: ScratchSpace {}

// Blanket implementation
impl<T: ScratchSpace> Workspace for T {}
