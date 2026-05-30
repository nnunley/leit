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
segment. Fields: `magic` (u32), `version` (u32), `format_flags` (u32),
`document_count` (u32, total documents in this segment), then the section offsets
**`field_table_offset`, `lexicon_offset`, `postings_table_offset`,
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

**Implementation note (ITER-0004 T2):** `SegmentHeader` (`crates/leit_index/src/segment_format/header.rs`)
uses explicit manual little-endian field (de)serialization (`u64::from_le_bytes`/`to_le_bytes`) rather than
a `bytemuck::from_bytes` Pod cast of the header struct. Two reasons: (1) the header is read exactly once per
segment open, so a zero-copy cast of its 80 bytes is immaterial — the real zero-copy surface is the SECTION
data, which the borrowed section readers (ITER-0004 T4, DEC-08) expose without copying; (2) manual
`from_le_bytes` is endianness-correct on every host, whereas a native-field `#[repr(C)]` Pod cast would be
wrong on big-endian and only the alignment-1 `[u8; N]`-byte-array Pod style (as the segment_ids types use)
would be portable — adding accessor verbosity for no functional gain on a read-once struct. The on-disk byte
layout is exactly as DEC-05 specifies (fixed LE, 80 bytes); only the in-memory access idiom differs.
DOWNSTREAM (ITER-0005 mmap): keep this manual decode — do NOT convert the header to a native Pod cast.
Also: the v1 magic is `b"LSG1"` (DISTINCT from legacy `b"LSEG"`) so legacy segments are cleanly rejected with
`BadMagic` rather than misread (DEC-16 reject-and-rebuild + STORY-0039 never-silent-corruption).

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

**Implementation note (ITER-0004 T3):** Footer is a 4-byte fixed-layout little-endian structure at `footer_offset` containing a single u32 CRC32C (Castagnoli, polynomial 0x1EDC6F41) checksum. The checksum covers all segment bytes from offset 0 up to (but not including) `footer_offset`, so it protects the header, all data sections, and block metadata. CRC32C was chosen over rapidhash (available in workspace dependencies) because rapidhash requires `std` and does not compile in `no_std+alloc` environments — leit_index must remain `no_std` compatible. CRC32C is deterministic, fast (bitwise loop per byte), and sufficient for detecting corruption. The checksum is computed via `compute_checksum()` and verified via `Footer::verify()`, called during `open_with_validation(Full)` in T6. No per-section checksums: a single segment-wide CRC32C is the minimal integrity check for v1 (DEC-10 rationale).

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
block-metadata section's v1 schema **must** carry, per block, at least a scorer-agnostic
impact upper bound (`max_term_freq` — see DEC-19; Phase 3 derives `max_score` from it at
query time) and the doc-range needed for WAND/MaxScore skipping, so Phase 3 can prune without
a format change. Recorded here so the ITER-0005 doc-range decision honors it.

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
| DEC-11 block boundary strategy | STORY-0087 | ITER-0002 (codec) / ITER-0004 (writer) | block codec conformance |
| DEC-12 v1 postings/block layout | STORY-0088 | ITER-0002 | SCENARIO-0006 |
| DEC-13 decode-scratch ownership | STORY-0079/0089 | ITER-0003 | SCENARIO-0019/0024 |
| DEC-14 TF/positions/payloads cursor | STORY-0080 | ITER-0003 | cursor trait tests |
| DEC-15 index→cursor integration | STORY-0001 | ITER-0003B | SCENARIO-0026 |
| DEC-16 Phase 1 deprecation | N/A | ITER-0004 | DirectorySegmentView shim |
| DEC-17 block-meta placement | STORY-0085/0022 | ITER-0005 T1/T2/T3 | schema + reader + writer |
| DEC-18 doc-range implicit | STORY-0086 | ITER-0005 T1/T2/T3/T4 | schema + round-trip |
| DEC-19 block-meta content schema | STORY-0023/0034 AC-1 | ITER-0005 T1/T2/T3/T4 | POD round-trip + cursor lowering |
| DEC-20 migration tooling + deprecation | STORY-0033 | ITER-0005 T8 | rewrite round-trip |
| DEC-21 mmap loading + thread-safety | STORY-0027/0037 | ITER-0005 T7 | SCENARIO-0018/0025 equivalence tests |

---

## DEC-11 — Postings block boundary: fixed document-count blocks (STORY-0087) — RESOLVED

**Decision:** v1 postings blocks are bounded by a **fixed document count** of **128 documents
per block** (the last block may be short). Boundaries are by doc-count, **not** fixed byte size
and **not** an adaptive/merge heuristic.

**Rationale (measured against ITER-0000 evidence):**
- The ITER-0000 wind-tunnel corpus is Zipfian (a few very long postings lists, a long tail of
  short ones). A **fixed byte-size** block would split a list at unpredictable doc positions,
  forcing partial-integer state across block boundaries and making per-block doc-range/skip
  metadata (the Phase 3 WAND constraint recorded for ITER-0005) awkward to compute. A fixed
  **doc-count** block gives every block a clean `[first_doc, last_doc]` range and a known item
  count, which is exactly what skip-lists and block-max metadata need.
- 128 docs/block is the conventional Lucene/PForDelta block size and matches the decode-batch
  granularity that keeps the decode scratch small and cache-resident. It is large enough to
  amortize per-block header cost over the long lists that dominate query latency in the baseline
  (`single_term` ≈ 136 µs/1k), small enough that selective block decode (the deferred ITER-0003
  `advance_to` skip) skips meaningful work.
- Short lists (< 128 docs — the Zipfian tail) occupy a single block; no padding, no waste.

**Consequence / forward-compat:** Each block carries its document count and first/last doc in its
header, so ITER-0005's block-metadata sidecar can reference blocks by ordinal and attach
`max_score` + doc-range without re-deriving boundaries. The boundary constant lives in one place
(`BLOCK_DOC_COUNT`) so a future codec may tune it without a format-break (the count is encoded
per block, not assumed by readers).

## DEC-12 — v1 postings/block layout for compressed traversal (STORY-0088) — RESOLVED

**Decision:** A postings list serializes as a sequence of independently-decodable **blocks**.
Each block is laid out as:

```
block := block_header doc_id_stream tf_stream
block_header := varint(doc_count) varint(first_doc) varint(last_doc) varint(doc_bytes_len)
doc_id_stream := varint(first_delta) varint(delta)*        # deltas from previous doc id
tf_stream     := varint(tf)*                                # one per doc, parallel to doc stream
```

- **Doc IDs** are **delta-encoded** then **LEB128 varint**-encoded (deltas are non-negative
  because postings are doc-sorted). The first doc in a block is stored as a delta from 0 (i.e. its
  absolute value), so each block is self-contained and decodable without the previous block
  (DEC-11 enables this).
- **Term frequencies** are LEB128 varint-encoded, one per doc, in a stream parallel to the doc-id
  stream — so a doc-only cursor can decode the doc stream and skip the TF stream using
  `doc_bytes_len`.
- **`doc_bytes_len`** in the header lets a reader locate the TF stream (and the next block) without
  decoding the doc stream — the basis for the deferred ITER-0003 selective/skip decode.
- **Codec marker:** a postings list is prefixed by a 1-byte `CodecId` (DEC: `0 = DeltaVarint`
  single-block, `1 = BlockDelta` doc-count blocks). This is the per-list marker; reserving the
  *field in the segment format* is deferred to ITER-0004 (STORY-0002 AC-3).

**Traversal-semantics contract (uncompressed ↔ compressed API stability — STORY-0088 AC-2 is the
e2e proof, deferred to ITER-0003):** the codec **API speaks named segment-resident types**
`SegmentLocalDocId` and `TermFreq` (the u32 values are delta-encoded and varint-encoded at the
byte level, unchanged). The generic `EntityId` is lowered to `SegmentLocalDocId` at the
segment-write boundary (ITER-0004). Decode produces exactly the `(SegmentLocalDocId, TermFreq)`
sequence, in doc-ascending order, that the in-memory `PostingsList<Id>` will hold — byte-for-byte
equal round trip (SCENARIO-0006). The codec writes decoded values into **caller-provided output
buffers** (`&mut Vec<SegmentLocalDocId>` and `&mut Vec<TermFreq>`); it does **not** own a
decode-scratch type. The decode-scratch ownership wrapper and the cursor adaptor over this API
are decided and built in ITER-0003 (STORY-0079) — the codec layer is deliberately
scratch-ownership-agnostic so that decision is not boxed in here.

**Decided in ITER-0002 against the ITER-0000 wind-tunnel baseline; enforced by the codec
implementation (SCENARIO-0006 round-trip) and measured by SCENARIO-0070 (codec comparison).**

## DEC-13 — Decode-scratch ownership model (STORY-0079, STORY-0089) — RESOLVED

**Decision:** **Workspace-borrowed** decode scratch. A `DecodeScratch { docs: Vec<SegmentLocalDocId>,
tfs: Vec<TermFreq> }` is owned by the *caller* (the query workspace / executor) and passed into cursor
construction by `&'a mut DecodeScratch`. A cursor borrows the scratch for its lifetime, decodes the
codec payload into it, and steps over it. The scratch is `clear()`ed (length reset, **capacity
retained**) and reused across sequential cursors, so steady-state traversal performs **zero heap
allocation** on the hot path.

**Rejected alternatives:**
- *Cursor-owned* (each cursor allocates its own `Vec`s): simplest lifetimes, but allocates per cursor —
  fails the zero-alloc hot-path target (STORY-0016) when a query opens many term cursors.
- *Pooled / thread-local*: best amortization but needs `std` (thread locals) or an allocator-coupled
  pool; violates `no_std` and adds reuse-complexity that the borrowed model already buys.

**Rationale:** the borrowed model gives explicit, compiler-checked lifetimes (no stale pointers,
no use-after-free — the scratch outlives every cursor that borrows it), is `no_std`+`alloc`-only, and
matches the scratch-ownership-agnostic codec decode API (DEC-12) — the codec already writes into
caller-provided `&mut Vec<…>` buffers, so `DecodeScratch` is exactly those two buffers behind a named
type. This **resolves the `TODO(ITER-0003)` at `codec.rs:153`**: the wrapper is `DecodeScratch`, owned
by the workspace, borrowed by cursors.

**Validation:** SCENARIO-0019 / SCENARIO-0024 — drive N sequential cursors over distinct postings views
through one `&mut DecodeScratch` and assert buffer **capacity is stable after warm-up** (no realloc),
i.e. allocation count matches the zero-per-cursor profile. `no_std`-friendly (capacity-stability proxy
for an allocation counter). Decided in ITER-0003 against the ITER-0000 wind-tunnel baseline.

## DEC-14 — TF / positions / payloads cursor capability: layered (STORY-0080) — RESOLVED

**Decision:** **Layered** cursor capabilities, not a unified trait. `DocCursor` (doc traversal:
`current_doc`, `advance`, `advance_to`) is the base; `TfCursor: DocCursor` adds `current_tf`;
`BlockCursor: DocCursor` adds block-summary access (`block_end_doc`, `block_max_score`). **Positions and
payloads are NOT in v1** — the v1 codec carries no positions (DEC-12). A future `PositionCursor: TfCursor`
is the reserved extension point; adding it does not change `DocCursor`/`TfCursor` callers.

**Rejected alternative:** a single unified `Cursor` trait exposing `tf()`, `positions()`, `payload()`
with `Option`/sentinel returns. Rejected because it forces every cursor (including a doc-only cursor that
should skip the TF stream entirely via `doc_bytes_len`, DEC-12) to carry capability it does not provide,
and pushes capability checks to runtime instead of the type system.

**Rationale:** layering keeps the simple/hot path (doc-only and doc+tf traversal) lean and
allocation-free, lets a scorer request exactly the capability it needs at the type level, and aligns with
the existing thin trait stack already in `leit_postings` (which this iteration evolves to the canonical
`advance_to`/`CursorStatus` API). Allocation-free guarantee: none of the layered methods allocate; only
`DecodeScratch` may grow, and only if a block exceeds current capacity (DEC-13).

**Validation:** crate-internal cursor integration tests exercise `DocCursor`/`TfCursor` over all
cursor-traversal query shapes (single / OR / AND / fielded / BM25F operands). Full *wired index*
query-path confirmation is subsumed by ITER-0003B's ranking-equivalence proof (SCENARIO-0026). Decided in
ITER-0003.

## DEC-15 — Index→cursor integration approach (STORY-0001, ITER-0003B) — RESOLVED

**Decision:** Dual cursor sources behind one trait-based executor. The `InMemoryIndex` term-scoring path
is refactored to score through a single helper generic over `leit_postings::cursor::TfCursor`. The DEFAULT
in-memory production path uses a new zero-copy in-memory cursor (`MemPostingsCursor`) over `&[PostingEntry]`
— no encode, no decode, no allocation. The COMPRESSED path (`PostingsView` + `DecodeScratch` +
`CompressedCursor`) is wired as an alternate execution source through the same machinery and proven to
produce identical top-k.

**Rationale:** The only concrete `TfCursor` impl, `CompressedCursor`, decodes codec-encoded bytes; routing
the in-memory path through it would force encode-once + decode-per-query — a perf regression for the
in-memory index, which already holds uncompressed postings in RAM. The segment format that will naturally
hold compressed bytes does not exist until ITER-0004. A zero-copy in-memory cursor keeps the default path
non-regressing while still unifying execution on the cursor trait API.

**Boxing-in:** None. ITER-0004 later supplies compressed segment bytes into the SAME `PostingsView`
/`CompressedCursor` source already wired here. The in-memory cursor lives in `leit_index` (it borrows the
`leit_index`-owned `PostingEntry`), so `leit_postings` stays unaware of `leit_index` (clean dependency
direction; orphan rule satisfied — `leit_index` owns the cursor type, `leit_postings` owns the trait).

**Evidence:** non-regression = all existing leit_index + integration tests stay green; ranking equivalence
= SCENARIO-0026 (in-memory vs DeltaVarint vs BlockDelta cursor sources yield bit-identical top-k).

## DEC-16 — Phase 1 segment format: DEPRECATE (frozen shim), then remove later (ITER-0004) — REVISED

**Revision (2026-05-30, user decision):** the original "delete in ITER-0004" stance below is SOFTENED to
**deprecate, don't delete**. The Phase 1 directory format is merged, upstream-accepted code (PR #1); rather
than remove it in this PR, it is kept as a **frozen, `#[deprecated]` shim** so external code still compiles
(with a deprecation warning) and legacy bytes remain readable. Concretely in ITER-0004 T7:
- The new DEC-05 view becomes the canonical `leit_index::SegmentView`; `InMemoryIndex::to_segment_bytes`
  emits the new format.
- The old directory reader is RENAMED to `DirectorySegmentView` and marked `#[deprecated]`; `SectionKind`
  stays exported, `#[deprecated]`. Both remain able to read legacy directory-format bytes (frozen — no
  further development). A minimal test keeps the shim exercised.
- A future release removes the shim. Downstream (ITER-0005 mmap, ITER-0006 merge) builds ONLY on the new
  format; the shim is not extended.
This keeps maintenance cost low (a frozen deprecated reader ≠ an actively-maintained dual path) while
respecting accepted upstream code. The ITER-0004 PR description must flag the deprecation for Bruce.

**Original decision (superseded by the revision above):** The DEC-05 fixed-header v1 segment format
**replaces** the pre-existing Phase 1 directory-based segment format (`crates/leit_index/src/segment.rs`:
`SegmentView`/`SectionKind` directory, u16 version, u32 offsets; writer `codec.rs::encode_segment` +
`InMemoryIndex::to_segment_bytes`). No dual-path reader, no parallel v1/v2 coexistence. The Phase 1
round-trip tests (`crates/leit_index/tests/segment_roundtrip.rs`, the segment portion of
`crates/leit_integration_tests/tests/phase1_readiness.rs`) are migrated to assert against the new format.

**Rationale:** leit is pre-1.0 with no production segments in existence — the Phase 1 format is a
test-only serialization of the in-memory index, never persisted by any consumer. A dual-path or
coexist strategy would add a second reader implementation, version-dispatch logic, and migration
tooling to preserve compatibility with data that does not exist (KISS/YAGNI). The handover specifies a
single clean versioned format with clean rejection of unknown versions, which `version: u32` + structured
`SegmentError::UnsupportedVersion` already deliver. If real persisted segments ever predate this change,
the correct recovery is an index rebuild, not a compatibility shim.

**Surfaced by:** ITER-0004 PAR scope review (both reviewers flagged the undefined Phase 1 relationship as
CRITICAL). Reviewer A leaned EVOLVE, Reviewer B leaned REPLACE; the orchestrator chose REPLACE on the
no-production-data + pre-1.0 grounds above and the absence of any compatibility obligation in the spec.

**Boxing-in:** None for downstream iterations. The replacement header is the complete DEC-05 layout
(block_meta/stored_fields/columnar/footer offsets all reserved and written now, pointing at empty
sections in v1-core), so ITER-0005 (block-meta content, mmap) and Phase 3 (columnar, stored fields)
fill reserved slots without a header rewrite.

**Verification:** ITER-0004 SCENARIO-0044 (write→read round-trip on the new format) + SCENARIO-0045
(unknown-version clean rejection); the migrated Phase 1 tests stay green against the new format; the
old directory format survives as a `#[deprecated] DirectorySegmentView` + `#[deprecated] SectionKind`
shim with a retained test proving it still reads a legacy directory buffer (deprecation, not deletion,
per the 2026-05-30 revision).

## DEC-17 — Block-meta physical placement: single grouped fixed-width table (STORY-0085, STORY-0022) — RESOLVED

**Decision:** The block-metadata sidecar is a **single grouped, fixed-width table** of
`BlockMetadataEntry` structs in the `block_meta` section, addressed by the header's
`block_meta_offset`. Each entry is 12 bytes (little-endian POD). A segment's postings blocks
are stored in postings order (all blocks for term 0, then all blocks for term 1, etc.),
so the table is contiguous and indexable.

**Rationale:** O(1) seek via `block_meta_offset + block_index * 12` (no per-block header
overhead, no variable-width encoding). Mmap-friendly (flat POD table, no heap pointers,
viewable directly from a memory-mapped region). Simplest ITER-0006 merge rebuild: stream
the new merged postings and recompute the block-summaries table sequentially into the new
segment. An alternative (adjacent block-summaries per-term, stored inline with postings
metadata) would require streaming access (slower for random block queries) and would bind
block metadata to postings-table layout, forcing a format change if postings encoding
evolves independently (e.g., per-term codec selection).

**Term→block mapping:** Each postings-table entry carries `first_block_index` and
`block_count` (fields to be added in ITER-0005 T3 writer). A term's blocks are the
contiguous range `[first_block_index .. first_block_index+block_count)` in the `block_meta`
table, accessed at offset `block_meta_offset + first_block_index*12`.

**Enforced by:** ITER-0005 T1 (schema definition), T2 (reader), T3 (writer).

## DEC-18 — Doc-range representation: implicit (re-derived from DEC-11 fixed 128-doc blocks) (STORY-0086) — RESOLVED

**Decision:** Block doc-ranges are **implicit**, re-derived from DEC-11's fixed 128-doc-per-block
boundaries. Each block's `end_doc` is stored (u32 LE, per the schema below), but `first_doc` is
NOT stored — it is re-derived.

**Derivation is PER-TERM, not global.** The block-meta table is segment-wide (blocks of all terms
stored contiguously), so the implicit rule MUST be scoped to a single term's block range — located
via `first_block_index` + `block_count` in that term's postings-table entry — never across a term
boundary:
- For block `i > 0` **within the same term**: `first_doc = (block[i-1].end_doc) + 1`.
- For block `0` of a term (the term's first block): its lower bound is the term's first posting
  doc-id. This is NOT a stored or table-derivable quantity — applying `prev.end_doc + 1` across the
  preceding term's last block would be WRONG. The skip/WAND algorithm does not need it: skipping
  uses only the monotonic per-block `end_doc` upper bounds (find the first block whose
  `end_doc >= target`). The exact `first_doc` of a term's first block, if ever needed, comes from
  decoding the first posting — it is deliberately not stored.

The final block of a term may be short (fewer than 128 docs); its `end_doc` is the actual last doc
in that term's postings, not padded.

**Overhead:** Implicit representation = 4 bytes per block (only `end_doc`). Explicit
representation (storing `(first_doc, last_doc)`) = 8 bytes per block. Implicit saves 50%
space and computes the same data with O(1) arithmetic.

**Forward compatibility with DEC-04:** The per-block bound (here, the doc-range) is
sufficient for Phase 3 WAND pruning. The format does NOT require padding short final blocks;
ITER-0006 merge may re-block postings with a different 128-doc alignment without violating
the format.

**Enforced by:** ITER-0005 T1 (schema definition), T2 (reader), T3 (writer), T4 (round-trip test).

## DEC-19 — Block-meta v1 content schema: three fixed-width fields, LE POD (STORY-0023, STORY-0034 AC-1) — RESOLVED

**Decision:** Each block-metadata entry is a **12-byte, little-endian, zero-copy POD** struct
with three fields (each u32 LE):

1. **`end_doc` (u32 LE):** Inclusive end document ID for this block. Per DEC-18, `first_doc`
   is implicit and re-derived. The range `[first_doc, end_doc]` spans all documents in the
   block.

2. **`max_term_freq` (u32 LE):** **Scorer-agnostic impact upper bound**, NOT a BM25 `score`.
   This is the maximum term frequency (raw, unscored) across all documents in the block.
   **Reconciliation of DEC-04's "max_score" wording:** a segment v1 carries NO scorer
   parameters (IDF, field weights, BM25 constants). Phase 3 WAND derives the actual query-time
   `max_score` from `max_term_freq` at execution time, using the query's scorer params, with
   **no segment format change** (the field remains `max_term_freq`). **Rejected alternative:**
   storing BM25 `max_score` directly would require embedding scorer params (IDF, field
   weights) in the segment, violating the scorer-agnostic principle (DEC-04); instead, the
   executor recomputes scores from the scorer-agnostic `max_term_freq` bound.

3. **`decode_offset` (u32 LE):** **Relative byte offset to the block's compressed payload**,
   relative to the term's postings-data section start (i.e., relative to
   `postings_data_offset + postings_data_offset_of_this_term`). This field is a segment-layer
   detail; it is NOT part of the cursor-layer `BlockSummary{end_doc, max_term_freq}` that
   Phase 3 WAND consumes. On ITER-0006 merge, the writer re-encodes postings into new blocks
   at new absolute positions and recomputes the relative `decode_offset` from the merged
   payload, with no format change (offsets remain relative).

   **Width justification (u32 vs the u64 of DEC-01):** a u32 relative offset bounds a single
   *term's* postings payload to 4 GiB. This is intentionally narrower than the u64 *segment*-level
   offsets (DEC-01): segment offsets span the whole file (can exceed 4 GiB), whereas a per-term
   payload realistically stays well under 4 GiB even for the most frequent term in a very large
   corpus. Keeping the per-block entry at 12 bytes (3×u32) halves block-meta overhead vs a u64
   offset. If a future corpus ever produces a single term exceeding 4 GiB of encoded postings,
   ITER-0006 merge tooling can split that term across multiple postings entries; the format does
   not break (the bound is a tooling constraint, not a hard format limit).

**Entry layout (12 bytes, POD bytemuck #[repr(C)]):**
```
offset  0-3:  end_doc (u32 LE)
offset  4-7:  max_term_freq (u32 LE)
offset  8-11: decode_offset (u32 LE)
```

**Cursor lowering (lowers to `BlockSummary`, ITER-0005 T4):** The segment's
`BlockMetadataEntry{end_doc, max_term_freq, decode_offset}` lowers into the cursor layer's
`BlockSummary{end_doc, max_term_freq}` when constructing a block-aware cursor. The
`decode_offset` is consumed only at the segment-reader level to locate compressed bytes;
the cursor itself sees only the summary (end_doc, max_term_freq).

**Enforced by:** ITER-0005 T1 (schema struct + round-trip bytemuck test),
T2 (BlockMetadataReader), T3 (writer), T4 (overhead-bytes proof + cursor lowering).

## DEC-21 — Memory-mapped segment loading: thread-safety model (STORY-0027, STORY-0037) — RESOLVED

**Decision:** Segment files may be loaded into memory via `MmapSegment::open(path)`, which
memory-maps the file and validates the header on construction. The mmap'd bytes are accessed
via `as_view() -> SegmentView<'_>`, with the lifetime tied to the mmap handle. The feature is
`mmap` (std-only, feature-gated); it is NOT included in no_std builds.

**Thread-safety model:** `memmap2::Mmap` is `Send + Sync` for read-only maps. A `MmapSegment`
owning the mmap handle can be wrapped in `Arc` and shared across threads. Each thread obtains
a `SegmentView<'_>` borrowing the mmap region with a lifetime tied to the borrowing thread's
scope. Rust's borrow-checking ensures no data races: the borrowed view cannot outlive either
the mmap handle or the thread's stack frame. **Phase 3 ITER-0006 merge may safely read source
segments in parallel**, wrapping `MmapSegment` in `Arc<MmapSegment>` and cloning the arc per
thread; each thread constructs local `SegmentView`s that cannot escape the thread's lifetime.

**Header validation:** On open, the header is validated (magic bytes and version checked) using
`ValidationMode::HeaderOnly` to minimize latency. Callers may request structural or full
validation via `as_view()`, which re-validates with the requested mode (offsets, section
ordering, checksum, etc.).

**Rejected alternatives:**
- **Lazy validation (defer to first access):** complicates error handling and delays failure
  detection. Eager validation on open (even lightweight header-only) catches I/O and corruption
  early.
- **Stored-in-memory snapshot:** defeats the purpose of mmap (zero-copy, large-file support).
  The borrowed `SegmentView` lifetime already provides safety without copying.

**Enforced by:** ITER-0005 T7 (MmapSegment + equivalence tests).
**Unblocks:** ITER-0006 (merge may read source segments in parallel via `Arc<MmapSegment>`).
