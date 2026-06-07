# Executable Index Design

## Summary

`leit_index` needs a common index-facing trait so `ExecutionWorkspace` can remain
the single public facade while supporting both the current `InMemoryIndex` and a
future segment-backed adapter.

The key constraint is architectural: the trait must describe the reusable search
substrate exposed by an index, not duplicate `ExecutionWorkspace`'s facade-level
methods. The facade owns planning scratch, query orchestration, and collector
coordination. The index trait supplies the retrieval facts and traversal
primitives that the facade consumes.

This document proposes:

- `PlanningIndex` for planner-facing lookup and field expansion
- `ExecutableIndex: PlanningIndex` for the full search substrate
- `SegmentIndex<'a>` as the execution adapter over `SegmentView<'a>`
- `ExecutionWorkspace` remaining the sole public plan/execute/search facade

## Problem

Today `ExecutionWorkspace` is concrete over `InMemoryIndex`.

That is enough for the current phase, but it does not match the intended shape
described elsewhere in the repo:

- `ExecutionWorkspace` is supposed to be reusable execution state
- `SegmentView` is supposed to be the canonical borrowed stored representation
- `leit_index` is supposed to be the integration boundary between projections,
  indexing, segment loading, and query execution

The missing piece is a common index-facing interface between the workspace and
the underlying index representation.

## Goals

- Keep `ExecutionWorkspace` as the only public facade for planning and execution
- Derive the new trait surface from the real search capabilities of
  `InMemoryIndex`
- Preserve the separation between storage/view types and execution types
- Avoid per-call allocation for field enumeration where practical
- Support a future segment-backed execution path without forcing `SegmentView`
  itself to become an execution object

## Non-Goals

- Do not make `SegmentView<'a>` directly executable
- Do not duplicate `ExecutionWorkspace` methods on the index trait
- Do not settle the final cursor abstraction in this document if a simpler
  borrowed postings/source seam is sufficient
- Do not change query semantics or scoring behavior

## Architectural Direction

The intended layering is:

```text
application entities
    -> projection
    -> index build / stored segment bytes
    -> executable index representation
    -> ExecutionWorkspace facade
    -> collectors / scored hits
```

In concrete terms:

- `InMemoryIndex` is one executable index representation
- `SegmentView<'a>` is the raw borrowed storage/view representation
- `SegmentIndex<'a>` is the executable adapter over `SegmentView<'a>`
- `ExecutionWorkspace` operates over index traits, not directly over a concrete
  storage type

This preserves the handoff's separation:

- storage format / bytes
- traversal and retrieval facts
- higher-level query execution orchestration

## Proposed Traits

### `PlanningIndex`

`PlanningIndex` is the minimal planner-facing interface.

It should cover:

- field name resolution
- term resolution
- available field enumeration
- default field enumeration for unfielded term expansion

Conceptually:

```rust
pub trait PlanningIndex: FieldRegistry + TermDictionary {
    fn for_each_field(&self, f: &mut dyn FnMut(FieldId));
    fn for_each_default_field(&self, f: &mut dyn FnMut(FieldId));
}
```

Notes:

- `FieldRegistry` and `TermDictionary` already exist in `leit_query`
- callback-based enumeration avoids forcing a fresh `Vec<FieldId>` allocation
- `search fields` are not part of this trait; they are derived per query during
  planning

### `ExecutableIndex`

`ExecutableIndex` extends `PlanningIndex` with the retrieval surface needed by
scoring and boolean evaluation.

Conceptually:

```rust
pub trait ExecutableIndex: PlanningIndex {
    fn document_count(&self) -> u32;

    fn field_stats(&self, field: FieldId) -> Option<FieldStatsView>;
    fn field_doc_length(&self, doc_id: u32, field: FieldId) -> u32;

    fn for_each_doc(&self, f: &mut dyn FnMut(u32));

    fn term_entry(&self, term: TermId) -> Option<TermEntryView<'_>>;

    fn open_postings(&self, term: TermId) -> Option<Self::Postings<'_>>;
    fn posting_block_summaries(&self, term: TermId) -> Option<Self::BlockSummaries<'_>>;

    type Postings<'a>
    where
        Self: 'a;

    type BlockSummaries<'a>
    where
        Self: 'a;
}
```

The exact associated types are open. The important point is that this trait
describes the index/search substrate, not a second facade.

## Borrowed Views

The trait should return small borrowed views rather than exposing full internal
maps.

For example:

```rust
pub struct FieldStatsView {
    pub field_id: FieldId,
    pub doc_count: u32,
    pub total_terms: u32,
}

pub struct TermEntryView<'a> {
    pub field_id: FieldId,
    pub term_id: TermId,
    pub term_text: &'a str,
}
```

This keeps the contract storage-agnostic and avoids leaking `InMemoryIndex`'s
internal representation into the trait.

## Why Not Put `execute_plan` on the Trait?

That approach was considered and rejected.

Putting facade-shaped methods like `execute_plan`, `search`, or other
workspace-like orchestration methods on the trait makes the abstraction too
high-level and duplicates the semantic role of `ExecutionWorkspace`.

The correct split is:

- the trait provides capabilities
- the workspace orchestrates those capabilities

If the trait mirrors the workspace, the architecture becomes muddy and future
segment-backed execution is harder to reason about.

## Why `SegmentIndex`, Not `SegmentView`

`SegmentView<'a>` should remain a storage/view type.

It is the canonical borrowed representation of a serialized segment, and it is
valuable precisely because it is:

- lightweight
- immutable
- storage-oriented
- easy to validate and mmap

Making it directly implement `ExecutableIndex` would blur the line between:

- a raw borrowed segment view
- an execution-ready index representation

`SegmentIndex<'a>` is the right adapter layer:

- `SegmentView<'a>` owns no execution policy
- `SegmentIndex<'a>` interprets the view as a searchable index
- `ExecutionWorkspace` talks to `SegmentIndex<'a>` through `ExecutableIndex`

## Extraction From Current `InMemoryIndex`

The proposed trait is grounded in the current `InMemoryIndex` behavior.

Planning today uses:

- `FieldRegistry`
- `TermDictionary`
- `default_fields()`

Execution and scoring today use:

- `document_count()`
- field statistics / average field length
- per-document field lengths
- postings per term
- posting blocks for pruning
- term-entry metadata for BM25F term-expansion aggregation
- the full document set for `NOT`/complement operations

Those are the capabilities the trait should expose, either directly or through
small borrowed view types.

## Migration Plan

### Step 1

Introduce `PlanningIndex` and make `ExecutionWorkspace::plan(...)` generic over
it.

This is the smallest safe seam because it only depends on:

- field lookup
- term lookup
- default-field enumeration

### Step 2

Refactor internal boolean/scoring evaluation to consume `ExecutableIndex`
capabilities rather than `InMemoryIndex` internals.

This may require introducing:

- a postings-source abstraction
- borrowed metadata views
- a stable way to iterate all docs for complement logic

### Step 3

Implement `ExecutableIndex` for `InMemoryIndex`.

This becomes the reference implementation and locks the trait surface to real,
working behavior rather than speculative design.

### Step 4

Add `SegmentIndex<'a>` over `SegmentView<'a>` and implement `ExecutableIndex`
for it.

At that point, `ExecutionWorkspace` can execute over both representations
through the same facade.

## Open Questions

- Should `for_each_field` be kept if field-name lookup is already covered by
  `FieldRegistry`?
  Current answer: yes, because the workspace may need enumeration without
  forcing reconstruction from other metadata.

- Should postings access be cursor-first or reader/view-first?
  Current answer: open. Cursor-first matches execution needs; reader/view-first
  may align better with `SegmentView`.

- Should `ExecutableIndex` include any fast-path hooks, or should those remain
  internal helper methods on implementations?
  Current answer: keep fast-path hooks internal until multiple implementations
  prove a stable common shape.

## Recommendation

Adopt the following design constraints:

- `ExecutionWorkspace` is the only public facade
- `PlanningIndex` is the minimal shared planning seam
- `ExecutableIndex: PlanningIndex` is the common search substrate
- `SegmentView<'a>` stays storage-only
- `SegmentIndex<'a>` becomes the execution adapter over `SegmentView<'a>`
- the trait surface is derived from `InMemoryIndex`'s real capabilities, not
  from facade duplication

That gives `leit_index` a common interface that matches the handoff's intended
shape without collapsing storage, traversal, and orchestration into the same
type.
