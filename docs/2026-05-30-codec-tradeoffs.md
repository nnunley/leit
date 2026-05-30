# Codec Tradeoff Analysis: DeltaVarint vs BlockDelta

**Status:** Evidence-based analysis from SCENARIO-0070 codec benchmarks (ITER-0002, STORY-0006 AC-3).

**Measurement:** `cargo bench -p leit_wind_tunnel_index --bench codec_compare` on the deterministic wind-tunnel corpus (SEED=42, 1K and 10K documents, Zipfian term distribution).

---

## Summary

Both codecs compress postings lists to ~25–27% of the uncompressed baseline (8 bytes per posting).

- **DeltaVarint**: single-block, varint-encoded deltas, ~2.03–2.05 bytes/posting
- **BlockDelta**: 128-doc blocks with per-block headers, ~2.10–2.19 bytes/posting

**Decode latency:**
- 1K corpus: DeltaVarint ~285 µs, BlockDelta ~297 µs (encode/decode times are comparable; BlockDelta slower due to per-block overhead)
- 10K corpus: DeltaVarint ~1.34 ms, BlockDelta ~1.48 ms

**Encode latency:**
- 1K corpus: DeltaVarint ~188 µs, BlockDelta ~311 µs (~1.65× slower)
- 10K corpus: DeltaVarint ~1.67 ms, BlockDelta ~2.54 ms (~1.52× slower)

---

## Tradeoff Rationale

### DeltaVarint: Decode speed, simplicity
- **Single stream**: no block metadata to parse, minimal decode latency.
- **Simplest codec**: delta encoding + varints, lowest complexity on the read path.
- **Trade-off**: no block structure means future block-aware features (selective decode, skip, WAND doc-range pruning) require full decode.
- **Encode cost**: low; varint encoding is linear and fast.

### BlockDelta: Block-aware future evolution
- **128-doc blocks**: each block independently decodable; enables Phase 3 features (selective block skip, WAND pruning with block-level doc ranges).
- **Per-block header overhead**: first_doc, last_doc, doc_bytes_len increase encoded size slightly vs DeltaVarint.
- **Trade-off**: decode is ~4–11% slower due to per-block header parsing; blocks do not improve memory footprint (compression ratio is similar).
- **Encode cost**: higher; block boundaries and per-block headers add work.

---

## Conclusion

**For Phase 2 (v1)**: DeltaVarint is sufficient and simpler; it achieves the same compression ratio with lower latency.

**For Phase 3+ (selective decode, block-aware WAND)**: BlockDelta's block structure is necessary to enable those features without full decode. The ~4–11% decode-latency cost is acceptable when the alternative is a format migration.

The benchmark confirms that **compression efficiency is not the differentiator**—both codecs perform similarly. The decision is **architectural**: DeltaVarint for speed/simplicity in v1, BlockDelta for extensibility in v2+.

**Current production choice**: DeltaVarint is the default; BlockDelta is implemented and tested in parallel for Phase 3 integration.
