# Phase 2 Architectural Decisions

**Status:** Decisions of record for ITER-0001. Each decision is *design-decidable
without wind-tunnel measurement*; the code that enforces it is implemented in the
iteration noted under "Enforced by" (the deferred `· deferred:ITER-NNNN` ACs).

**Grounding:** `docs/leit_kernel_handover.md` §"Segment Architecture" (the format
sketch, versioning bias, and Open Questions) and the ITER-0001 serialization choice
(bytemuck zero-copy, little-endian — see `docs/superpowers/iterations/requirements/EPIC-009.md`).

**Cross-cutting premise:** segment-resident structures are **zero-copy POD** —
`#[repr(transparent)]`/`#[repr(C)]` over little-endian byte fields, viewed in place
from an mmap'd `&[u8]` via bytemuck, with no deserialization or heap reconstruction
on the read path. Every decision below is consistent with that premise.

---

## DEC-01 — Offset width: u64 (STORY-0043) — RESOLVED

**Decision:** Segment offsets are **unsigned 64-bit, little-endian**, absolute from
segment start. There is **no practical segment-size cap**. (The handover sketch used
`u32` and flagged the choice as open; the human chose u64 to remove any future
format-break-for-size risk — see Phase 3 forward-compatibility.)

**Rationale:** u64 offsets future-proof the format against large single segments
(stored fields, columnar, large postings) with no v2 migration ever needed for size.
The cost is a larger header (8-byte offsets, ~84 bytes vs ~44), negligible against
segment size and read once per open. Doc IDs remain segment-local **u32**
(`SegmentLocalDocId`): byte offsets are u64 (file positions may be large), but a single
segment is bounded to 2³² docs — an independent, generous limit that does not interact
with offset width.

**Verification:** documented here; `SegmentHeader` uses the u64 offset type (enforced
ITER-0004, STORY-0043 AC-3). STORY-0043 AC-2 (32-bit cap documentation) is N/A — u64
was chosen, so there is no cap to document.

**Enforced by:** ITER-0004.

## DEC-02 — Metadata tables: fixed-width entries; variable-width only for term bytes (STORY-0044)

**Decision:** `field_table` and `postings_table` use **fixed-width entries** so a
reader seeks to entry *i* with O(1) offset arithmetic and views the whole table as a
zero-copy `&[Entry]` (bytemuck slice cast). The **term dictionary** stores the
variable-length term bytes in a blob, addressed by a **fixed-width offset/length index**
(itself a POD table) → O(1) access to any term's bytes without scanning.

**Rationale:** Handover bias: "direct section lookup", "O(1) offset computation".
Fixed-width tables are exactly the zero-copy POD slices bytemuck gives us. Only term
strings are inherently variable; isolating them behind a fixed-width index keeps every
*metadata* access O(1) while paying variable-width cost only for the term bytes.

**Enforced by:** ITER-0004 (STORY-0044 AC-2/3).

## DEC-03 — Dictionary/postings coupling: separately addressable (STORY-0045)

**Decision:** The term dictionary (lexicon) and the postings metadata table are
**separate sections** with independent header offsets (`lexicon_offset`,
`postings_table_offset`), as in the handover sketch. A lexicon entry yields an index
into the postings table; the two are not interleaved.

**Rationale:** Separate sections can be validated, mmap'd, and evolved independently,
and keep each section a homogeneous POD table (DEC-02). Interleaving would mix
variable-width term bytes with fixed-width postings metadata, defeating O(1) seeks.
The handover sketch already gives them separate offsets.

**Enforced by:** ITER-0004 (STORY-0045 AC-2).

## DEC-04 — Mmap readiness scope (STORY-0046)

**Decision:** **mmap-friendly in v1** (plain little-endian POD, no heap pointers, no
relocations, viewable zero-copy): segment header, field table, term dictionary
(index + bytes), postings metadata table, postings data blocks, block metadata.
**Deferred / build-time or in-memory only in v1:** optional stored-fields section and
optional columnar section (their slots are reserved in the header but their v1 content
is minimal/empty — full content is Phase 3).

**Rationale:** The hot read path (header → section tables → postings → blocks) must be
zero-copy mmap for the performance goals. Stored/columnar are optional and off the hot
retrieval path, so they can lag without constraining v1. Reserving header slots now
(DEC-05) keeps the format forward-compatible.

**Enforced by:** ITER-0004 (STORY-0046 AC-2, SCENARIO-0047), ITER-0005 (mmap loading).

## DEC-05 — Segment header / offset strategy (STORY-0090)

**Decision:** A **fixed-layout, little-endian POD header** (bytemuck `Pod`,
alignment-1 byte-field layout consistent with the ID types) as the first bytes of the
segment. Fields: `magic` (u32), `version` (u32), `format_flags` (u32), then the
section offsets **`field_table_offset`, `lexicon_offset`, `postings_table_offset`,
`postings_data_offset`, `block_meta_offset`, `stored_fields_offset`,
`columnar_offset`, `footer_offset` (all u64 LE, per DEC-01)**. **Offsets are absolute
from segment start.**
Endianness is fixed little-endian on every host. A trailing **footer** carries the
optional checksum (DEC-10).

**Rationale:** Extends the handover sketch (adds `magic`, `format_flags`,
`stored_fields_offset`, `footer_offset`). Absolute offsets are the cheapest to
validate — each must be `<= segment_len` and sections must be non-overlapping/ordered
— and they let a reader jump directly to any section. `format_flags` marks which
optional sections are present (DEC-10). Little-endian POD = the bytemuck zero-copy
premise.

**Enforced by:** ITER-0004 (STORY-0090 AC-2, SCENARIO-0025).

## DEC-06 — Block-aware capability scope (STORY-0081)

**Decision:** Block-aware accessors (`block_max_score()`, `block_end_doc()`, block
boundaries) live on a **dedicated public trait** (e.g. `BlockCursor`) that is
**separate from** the base cursor trait — public so the format's block metadata is
reachable and a future WAND/MaxScore scorer can consume it, but not forced on basic
cursor users. **v1 exposes the data/API surface only; block-skipping *execution*
(WAND pruning) is Phase 3.**

**Rationale:** Handover makes block metadata "a first-class citizen". Exposing it via a
separate trait honors the architectural split (traversal API vs execution) without
committing v1 to the pruning algorithm. Keeping it off the base trait avoids boxing in
non-block cursors.

**Verification:** trait visibility/boundary enforced in code (ITER-0003,
STORY-0081 AC-2). **RESOLVED — public dedicated `BlockCursor` trait (confirmed).**

**Enforced by:** ITER-0003.

## DEC-07 — Segment validation strategy: three lazy modes (STORY-0082)

**Decision:** A `ValidationMode` enum with **`HeaderOnly`**, **`Structural`** (default),
and **`Full`**:
- `HeaderOnly` — verify magic + version (reject unsupported) + header self-consistency. Cheapest open.
- `Structural` — additionally verify every offset is in-bounds and sections are ordered/non-overlapping. The default for `SegmentView::open()`.
- `Full` — additionally verify the footer checksum (DEC-10) and per-section structural invariants.

Validation is otherwise **lazy**: per-access reads use bytemuck `try_*` casts, which
bounds-check length at the point of use regardless of mode, so traversal is always
memory-safe even under `HeaderOnly`.

**Rationale:** Balances startup latency against safety (handover: "easy validation of
offset ranges"). `Structural` is cheap (a handful of comparisons) and catches the
common corruption/truncation cases, so it's the safe default; `Full` is opt-in for
untrusted inputs. Because `try_cast` validates every slice access anyway, even the
cheapest mode cannot cause UB.

**Verification:** `ValidationMode` + `open_with_validation()` (ITER-0004,
STORY-0082 AC-2, SCENARIO-0020). **RESOLVED — three modes; `Structural` default; `Full`
adds the DEC-10 footer checksum (not surfaced for change; recorded default stands).**

**Enforced by:** ITER-0004.

## DEC-08 — View borrowing model: fully borrowed (STORY-0083)

**Decision:** All segment section views **borrow `&[u8]` directly** (zero-copy). View
types hold a byte slice plus validated offsets/lengths; there are **no eagerly-decoded
owned fields**. The only work at view-construction time is bounds validation (returns
indices, not copies).

**Rationale:** Directly follows the bytemuck zero-copy premise and the handover's
"low-cost borrowed views" + minimal-allocation principle. Any eager copy would
reintroduce allocation on the hot path. Decoded values (e.g. a `u32` from an offset
table) are produced on demand by cheap LE reads, not cached.

**Enforced by:** ITER-0004 (STORY-0083 AC-2).

## DEC-09 — Builder vs read-only type separation: strict (STORY-0084)

**Decision:** **Strict separation.** Writer/builder types (`SegmentBuilder` and
friends) that accumulate and serialize a segment are **distinct** from the read-only
view types (`SegmentView` and section views). Builder types never appear in query/read
paths; view types are never mutable post-construction.

**Rationale:** Segments are immutable; conflating builder and reader invites misuse
(e.g. mutating a "view"). The handover's clean split — on-disk representation ↔
traversal views ↔ orchestration — is naturally expressed as writer-produces-bytes,
reader-views-bytes. Strict separation also lets the reader stay `no_std`/zero-copy
while the builder may freely allocate.

**Enforced by:** ITER-0004 (STORY-0084 AC-2).

## DEC-10 — Versioning & checking scope (STORY-0047)

**Decision (minimal but real):**
- **Version:** `version: u32` in the header; readers **reject unsupported (future) versions cleanly** with a structured error. Backward compatibility is promised for a *bounded* set of versions; dropping support means rebuild-via-tooling (handover model).
- **Feature flags:** `format_flags: u32` bitfield marks which optional sections are present (stored_fields, columnar), so a reader knows whether those offsets are meaningful.
- **Checksum:** a **single footer checksum** over the segment body (algorithm: a fast non-cryptographic hash — candidate: the same rapidhash family already vendored, or crc32c). Validated only in `Full` mode (DEC-07). **No per-section checksums in v1.**
- **Magic:** a 4-byte magic constant as the first header field for quick format identification.

**Rationale:** Handover wants explicit versioning + clean rejection from the start, but
warns against over-engineering. Version + flags + one optional checksum + magic is the
minimal set that supports clean rejection, optional-section detection, and integrity
checking, without the cost/complexity of per-section checksums.

**Verification:** version-rejection + flag handling + checksum (ITER-0004,
STORY-0047 AC-2/3). **RESOLVED — include a single footer checksum in v1 (confirmed);
algorithm finalized in ITER-0004 (rapidhash-family or crc32c).**

**Enforced by:** ITER-0004.

---

## Decisions epic anchor (STORY-0078)

All eight-plus must-resolve decisions above are recorded with a decision, a rationale,
and a verification method (the enforcing iteration + AC/scenario). Dependent
implementation (ITER-0002 codecs, ITER-0003 cursors, ITER-0004 segment format) may now
proceed against a stable decision point. Decisions flagged **[SURFACE]** were raised to
the human for confirmation; the recorded choice is the confirmed/default position.

## Phase 3 forward-compatibility (does any v1 decision box in Phase 3?)

Phase 3 adds **WAND / MaxScore block-skipping execution**, **columnar field content**,
and **real-corpus loading**. The v1 decisions are designed to *enable* these without a
format break:

1. **Block metadata is first-class and mmap-friendly in v1** (DEC-04), even though
   block-skipping *execution* is Phase 3 (DEC-06). The per-block `max_score` and doc
   ranges live in the format now, so Phase 3 WAND reads them with **no format change** —
   this is the handover's explicit "first-class, not bolted on later" intent.
2. **The block-aware API surface ships in v1** as a dedicated `BlockCursor` trait
   (DEC-06) and the MaxScoreScorer block-structure contract (ITER-0003, SCENARIO-0002).
   Phase 3 pruning is a new *consumer* of an existing surface, not an API break.
3. **Optional-section slots are reserved now** (DEC-05 header carries
   `stored_fields_offset`/`columnar_offset`; DEC-10 `format_flags` marks presence).
   Phase 3 fills the columnar slot and sets its flag — **no version bump required**;
   old readers see the flag clear and ignore it.
4. **Explicit versioning + rebuild path** (DEC-01, DEC-10): any change that *cannot* be
   done via a reserved slot is a clean `version` bump with clean rejection by old
   readers and a tooling rebuild — never in-hot-path legacy support.
5. **Zero-copy bytemuck POD** generalizes: Phase 3's columnar and block structures are
   also flat POD tables, so the same `try_*` validated casts apply. No serialization
   rework.

**No segment-size limit:** DEC-01 was resolved to **u64 offsets**, so the previously
identified 4 GiB cap is gone — there is no v1 decision that forces a future
format-break for segment size. The only residual bound is `SegmentLocalDocId` = u32
(≤ 2³² docs per segment), which is independent of offset width and far beyond any
realistic single-segment doc count; it does not constrain Phase 3.

**Forward constraint placed on ITER-0005 (block-metadata schema, STORY-0086):** the
block-metadata section's v1 schema **must** carry, per block, at least `max_score` and
the doc-range needed for WAND/MaxScore skipping, so Phase 3 can prune without a format
change. Recorded here so the ITER-0005 doc-range decision honors it.

**Completed work check:** ITER-0000 (wind tunnel) is additive measurement infra and
boxes nothing; its STORY-0105 real-corpus path is explicitly a Phase 3+ plug-in against
stable output types. ITER-0001 ID types (`BlockId`, etc.) are precisely what Phase 3
WAND consumes — defining them now *helps* Phase 3.

## Decision → enforcement traceability

| Decision | Story | Enforced by | Proof |
|---|---|---|---|
| DEC-01 offset width u64 | STORY-0043 | ITER-0004 | header uses u64 offset type |
| DEC-02 fixed-width tables | STORY-0044 | ITER-0004 | O(1) entry seek |
| DEC-03 separate sections | STORY-0045 | ITER-0004 | independent offsets |
| DEC-04 mmap scope | STORY-0046 | ITER-0004/0005 | SCENARIO-0047 |
| DEC-05 header/offset strategy | STORY-0090 | ITER-0004 | SCENARIO-0025 |
| DEC-06 block-aware trait | STORY-0081 | ITER-0003 | trait visibility |
| DEC-07 validation modes | STORY-0082 | ITER-0004 | SCENARIO-0020 |
| DEC-08 fully borrowed views | STORY-0083 | ITER-0004 | view accessors |
| DEC-09 strict builder/reader | STORY-0084 | ITER-0004 | type boundary |
| DEC-10 versioning/checks | STORY-0047 | ITER-0004 | version-rejection |
