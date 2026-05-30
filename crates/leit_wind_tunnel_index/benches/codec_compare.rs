// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Codec comparison benchmarks: encode/decode latency and compressed size.
//!
//! This benchmark indexes a deterministic wind-tunnel corpus and extracts all postings,
//! then measures and reports:
//! - Encode time per codec
//! - Decode time per codec
//! - Compressed size ratio vs 8-byte uncompressed baseline
//!
//! Run with `cargo bench -p leit_wind_tunnel_index --bench codec_compare`.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented public `benches` fn"
)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use leit_core::{FieldId, SegmentLocalDocId, TermFreq};
use leit_index::InMemoryIndexBuilder;
use leit_postings::codec::{BlockDeltaCodec, Codec, CodecId, DeltaVarintCodec};
use leit_text::{Analyzer, FieldAnalyzers, UnicodeNormalizer, WhitespaceTokenizer};
use leit_wind_tunnel::{CorpusGenerator, corpus::GeneratedDoc};

/// Fixed seed so every benchmark run indexes byte-identical corpora.
const SEED: u64 = 42;
/// `title` field, matching the corpus generator's field 1.
const TITLE: FieldId = FieldId::new(1);
/// `body` field, matching the corpus generator's field 2.
const BODY: FieldId = FieldId::new(2);

/// Postings list type: `Vec` of (`SegmentLocalDocId`, `TermFreq`) tuples per term.
type PostingsList = Vec<Vec<(SegmentLocalDocId, TermFreq)>>;

/// Build the field analyzers used for indexing (whitespace tokenizer + unicode
/// normalizer on both fields), matching the wind-tunnel integration tests.
fn make_analyzers() -> FieldAnalyzers {
    let mut analyzers = FieldAnalyzers::new();
    analyzers.set(
        TITLE,
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new()),
    );
    analyzers.set(
        BODY,
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new()),
    );
    analyzers
}

/// Index a corpus and extract all postings as `Vec<(doc_id, term_freq)>` tuples.
fn extract_postings(corpus: &[GeneratedDoc]) -> PostingsList {
    let mut builder = InMemoryIndexBuilder::new(make_analyzers());
    builder.register_field_alias(TITLE, "title");
    builder.register_field_alias(BODY, "body");

    for doc in corpus {
        builder
            .index_document(
                doc.id,
                &[(TITLE, doc.title.as_str()), (BODY, doc.body.as_str())],
            )
            .expect("indexing should succeed");
    }

    let index = builder.build_index();

    // Extract all postings as (SegmentLocalDocId, TermFreq) tuples.
    // Postings are already doc-sorted from the index.
    let mut all_postings = Vec::new();
    for posting_list in index.postings_by_term().values() {
        let postings: Vec<(SegmentLocalDocId, TermFreq)> = posting_list
            .iter()
            .map(|p| (SegmentLocalDocId::new(p.doc_id), TermFreq::new(p.term_freq)))
            .collect();
        if !postings.is_empty() {
            all_postings.push(postings);
        }
    }
    all_postings
}

/// Measure compression ratio and emit a summary table.
fn report_compression_detailed(
    corpus_size: u32,
    total_postings: usize,
    codec_sizes: &[(&str, usize)],
) {
    // Baseline: 8 bytes per posting (u32 doc_id + u32 term_freq).
    let uncompressed_baseline = total_postings * 8;

    eprintln!(
        "\n=== Codec Compression Summary (corpus: {} docs, {} total postings) ===",
        corpus_size, total_postings
    );
    eprintln!("Baseline (uncompressed): 8 bytes per posting");
    eprintln!("  Total uncompressed: {} bytes", uncompressed_baseline);

    for (name, bytes) in codec_sizes {
        let ratio = if uncompressed_baseline > 0 {
            (*bytes as f64) / (uncompressed_baseline as f64) * 100.0
        } else {
            0.0
        };
        let avg_bytes_per_posting = if total_postings > 0 {
            *bytes as f64 / total_postings as f64
        } else {
            0.0
        };
        eprintln!(
            "  {}: {} bytes ({:.1}% of baseline, {:.2} bytes/posting)",
            name, bytes, ratio, avg_bytes_per_posting
        );
    }
}

fn bench_codec_encode_decode(c: &mut Criterion) {
    let generator = CorpusGenerator::new(SEED);

    // Prepare corpora and postings once outside the benchmark loop.
    #[expect(
        clippy::useless_vec,
        reason = "vec literal is the idiomatic way to construct this list"
    )]
    let corpora = vec![
        (1_000_u32, generator.generate(1_000)),
        (10_000_u32, generator.generate(10_000)),
    ];

    let all_postings: Vec<(u32, PostingsList)> = corpora
        .iter()
        .map(|(size, corpus)| (*size, extract_postings(corpus)))
        .collect();

    // Report compression sizes upfront.
    for (corpus_size, postings_list) in &all_postings {
        let mut total_postings = 0;
        let mut sizes = Vec::new();

        // Encode all postings with each codec and measure total size.
        let delta_varint_codec = DeltaVarintCodec;
        let block_delta_codec = BlockDeltaCodec;

        let mut dv_total_size = 0;
        let mut bd_total_size = 0;

        for postings in postings_list {
            if !postings.is_empty() {
                total_postings += postings.len();

                let dv_encoded = delta_varint_codec.encode(postings);
                dv_total_size += dv_encoded.len();

                let bd_encoded = block_delta_codec.encode(postings);
                bd_total_size += bd_encoded.len();
            }
        }

        sizes.push(("DeltaVarint", dv_total_size));
        sizes.push(("BlockDelta", bd_total_size));

        report_compression_detailed(*corpus_size, total_postings, &sizes);
    }

    let mut group = c.benchmark_group("codec_encode");
    for (corpus_size_label, corpus_size, postings_list) in all_postings.iter().map(|(s, p)| {
        let label = if *s == 1_000 { "1k" } else { "10k" };
        (label, *s, p)
    }) {
        let delta_varint_codec = DeltaVarintCodec;
        let block_delta_codec = BlockDeltaCodec;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("deltavarint/{}", corpus_size_label)),
            &corpus_size,
            |b, _| {
                b.iter(|| {
                    let mut total_bytes = 0;
                    for postings in postings_list {
                        if !postings.is_empty() {
                            let encoded = delta_varint_codec.encode(postings);
                            total_bytes += encoded.len();
                        }
                    }
                    criterion::black_box(total_bytes);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("blockdelta/{}", corpus_size_label)),
            &corpus_size,
            |b, _| {
                b.iter(|| {
                    let mut total_bytes = 0;
                    for postings in postings_list {
                        if !postings.is_empty() {
                            let encoded = block_delta_codec.encode(postings);
                            total_bytes += encoded.len();
                        }
                    }
                    criterion::black_box(total_bytes);
                });
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("codec_decode");
    for (corpus_size_label, corpus_size, postings_list) in all_postings.iter().map(|(s, p)| {
        let label = if *s == 1_000 { "1k" } else { "10k" };
        (label, *s, p)
    }) {
        let delta_varint_codec = DeltaVarintCodec;
        let block_delta_codec = BlockDeltaCodec;

        // Pre-encode for decode benchmarks.
        let dv_encoded: Vec<Vec<u8>> = postings_list
            .iter()
            .map(|p| {
                if p.is_empty() {
                    vec![CodecId::DeltaVarint.to_u8()]
                } else {
                    delta_varint_codec.encode(p)
                }
            })
            .collect();

        let bd_encoded: Vec<Vec<u8>> = postings_list
            .iter()
            .map(|p| {
                if p.is_empty() {
                    vec![CodecId::BlockDelta.to_u8()]
                } else {
                    block_delta_codec.encode(p)
                }
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("deltavarint/{}", corpus_size_label)),
            &corpus_size,
            |b, _| {
                b.iter(|| {
                    let mut decoded_count = 0;
                    for (encoded, original_postings) in dv_encoded.iter().zip(postings_list) {
                        let mut docs = Vec::new();
                        let mut tfs = Vec::new();
                        delta_varint_codec
                            .decode(encoded, &mut docs, &mut tfs)
                            .expect("decode should succeed");

                        // Sanity gate: verify decode matches original (compare typed values).
                        assert_eq!(docs.len(), original_postings.len(), "doc count mismatch");
                        assert_eq!(tfs.len(), original_postings.len(), "tf count mismatch");
                        for (i, (doc_id, tf)) in original_postings.iter().enumerate() {
                            assert_eq!(docs[i], *doc_id, "doc mismatch at index {i}");
                            assert_eq!(tfs[i], *tf, "tf mismatch at index {i}");
                        }

                        decoded_count += docs.len();
                    }
                    criterion::black_box(decoded_count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("blockdelta/{}", corpus_size_label)),
            &corpus_size,
            |b, _| {
                b.iter(|| {
                    let mut decoded_count = 0;
                    for (encoded, original_postings) in bd_encoded.iter().zip(postings_list) {
                        let mut docs = Vec::new();
                        let mut tfs = Vec::new();
                        block_delta_codec
                            .decode(encoded, &mut docs, &mut tfs)
                            .expect("decode should succeed");

                        // Sanity gate: verify decode matches original (compare typed values).
                        assert_eq!(docs.len(), original_postings.len(), "doc count mismatch");
                        assert_eq!(tfs.len(), original_postings.len(), "tf count mismatch");
                        for (i, (doc_id, tf)) in original_postings.iter().enumerate() {
                            assert_eq!(docs[i], *doc_id, "doc mismatch at index {i}");
                            assert_eq!(tfs[i], *tf, "tf mismatch at index {i}");
                        }

                        decoded_count += docs.len();
                    }
                    criterion::black_box(decoded_count);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_codec_encode_decode);
criterion_main!(benches);
