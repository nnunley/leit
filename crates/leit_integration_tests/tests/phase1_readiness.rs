// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Phase 1 readiness integration test suite.
//!
//! This suite validates the fundamental Phase 1 seams described in the
//! handoff docs and exposed by the current workspace crates.

use leit_collect::{Collector, CountCollector, TopKCollector};
use leit_core::{EntityId, Score, ScoredHit, ScratchSpace, Workspace};
use leit_fusion::{RankedResult, fuse_default};
use leit_index::{
    ExecutionWorkspace, InMemoryIndexBuilder, NoFilter, SearchScorer, SectionKind, SegmentView,
};
use leit_query::{
    FeatureSet, FieldRegistry, Planner, PlannerScratch, PlanningContext, QueryNode, TermDictionary,
};
use leit_score::{Bm25FScorer, Bm25Params, Bm25Scorer, FieldStats, ScoringStats};
use leit_text::{Analyzer, FieldAnalyzers, UnicodeNormalizer, WhitespaceTokenizer};
use proptest::collection::vec;
use proptest::prelude::*;

const F32_EPSILON: f32 = 1.0e-6;
const F64_EPSILON: f64 = 1.0e-12;

const fn u32_to_f32(value: u32) -> f32 {
    value as f32
}

fn assert_close_f32(actual: f32, expected: f32, context: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= F32_EPSILON,
        "{context}: expected {expected:.8}, got {actual:.8} (|delta|={delta:.8})"
    );
}

fn assert_close_f64(actual: f64, expected: f64, context: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= F64_EPSILON,
        "{context}: expected {expected:.12}, got {actual:.12} (|delta|={delta:.12})"
    );
}

fn bm25_reference(stats: &ScoringStats, params: Bm25Params) -> f32 {
    if stats.term_frequency == 0 || stats.doc_count == 0 {
        return 0.0;
    }

    let tf = u32_to_f32(stats.term_frequency);
    let doc_length = u32_to_f32(stats.doc_length);
    let avg_doc_length = stats.avg_doc_length.max(F32_EPSILON);
    let doc_count = u32_to_f32(stats.doc_count);
    let doc_frequency = u32_to_f32(stats.doc_frequency);
    let idf = ((doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1.0).ln();
    let dl_norm = 1.0 - params.b + params.b * (doc_length / avg_doc_length);
    let tf_sat = (tf * (params.k1 + 1.0)) / (tf + params.k1 * dl_norm);

    idf * tf_sat
}

fn bm25f_reference(
    fields: &[FieldStats],
    avg_doc_length: f32,
    doc_count: u32,
    doc_frequency: u32,
    params: Bm25Params,
) -> f32 {
    if doc_count == 0 {
        return 0.0;
    }

    let mut weighted_tf = 0.0_f32;
    let mut weighted_doc_length = 0.0_f32;

    for field in fields {
        weighted_tf += u32_to_f32(field.term_frequency) * field.weight;
        weighted_doc_length += u32_to_f32(field.field_length) * field.weight;
    }

    if weighted_tf == 0.0 {
        return 0.0;
    }

    let avg_doc_length = avg_doc_length.max(F32_EPSILON);
    let doc_count = u32_to_f32(doc_count);
    let doc_frequency = u32_to_f32(doc_frequency);
    let idf = ((doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1.0).ln();
    let dl_norm = 1.0 - params.b + params.b * (weighted_doc_length / avg_doc_length);
    let tf_sat = (weighted_tf * (params.k1 + 1.0)) / (weighted_tf + params.k1 * dl_norm);

    idf * tf_sat
}

const fn entity_roundtrip<Id: EntityId>(id: Id) -> ScoredHit<Id> {
    ScoredHit::perfect(id)
}

#[test]
fn test_entity_id_fundamentals() {
    let hit_u32 = entity_roundtrip(42_u32);
    let hit_u64 = entity_roundtrip(42_u64);

    assert_eq!(hit_u32.id, 42);
    assert_eq!(hit_u64.id, 42);
}

#[test]
fn test_score_fundamentals() {
    assert_close_f32(Score::new(0.85).as_f32(), 0.85, "score stores value");
    assert_close_f32(
        Score::new(1.5).as_f32(),
        1.5,
        "score preserves values above one",
    );
    assert_close_f32(
        Score::new(-0.5).as_f32(),
        -0.5,
        "score preserves negative values",
    );
    assert_close_f32(Score::ZERO.as_f32(), 0.0, "ZERO constant");
    assert_close_f32(Score::ONE.as_f32(), 1.0, "ONE constant");

    let weight = Score::new(2.5);
    assert_close_f32(weight.as_f32(), 2.5, "score preserves finite values");
}

#[test]
#[should_panic(expected = "score must be finite")]
fn test_score_rejects_nan() {
    let _ = Score::new(f32::NAN);
}

#[test]
fn test_score_arithmetic() {
    let a = Score::new(0.3);
    let b = Score::new(0.4);

    assert_close_f32((a + b).as_f32(), 0.7, "score addition");
    assert_close_f32((b - a).as_f32(), 0.1, "score subtraction");
    assert_close_f32((a * 2.0).as_f32(), 0.6, "score scalar multiplication");

    let mut sum = Score::new(0.5);
    sum += Score::new(0.2);
    assert_close_f32(sum.as_f32(), 0.7, "score add assign");

    let mut diff = Score::new(0.8);
    diff -= Score::new(0.3);
    assert_close_f32(diff.as_f32(), 0.5, "score sub assign");
}

#[test]
fn test_scored_hit_fundamentals() {
    let hit = ScoredHit::new(42_u32, Score::new(0.85));
    let perfect = ScoredHit::perfect(1_u32);
    let zero = ScoredHit::zero(2_u32);

    assert_eq!(hit.id, 42);
    assert_close_f32(hit.score.as_f32(), 0.85, "hit stores score");
    assert!(!hit.is_zero());
    assert_close_f32(perfect.score.as_f32(), 1.0, "perfect hit");
    assert_close_f32(zero.score.as_f32(), 0.0, "zero hit");
    assert!(zero.is_zero());
}

#[test]
fn test_scored_hit_ordering() {
    let high = ScoredHit::new(1_u32, Score::new(0.9));
    let mid = ScoredHit::new(2_u32, Score::new(0.5));
    let low = ScoredHit::new(3_u32, Score::new(0.1));
    let a = ScoredHit::new(1_u32, Score::new(0.5));
    let b = ScoredHit::new(2_u32, Score::new(0.5));

    assert!(high > mid);
    assert!(mid > low);
    assert!(high > low);
    assert_eq!(a.cmp(&b), core::cmp::Ordering::Less);
}

#[test]
fn test_bm25_known_answer() {
    let scorer = Bm25Scorer::new();
    let stats = ScoringStats {
        term_frequency: 2,
        doc_length: 100,
        avg_doc_length: 150.0,
        doc_count: 1000,
        doc_frequency: 50,
        ..ScoringStats::new()
    };

    let actual = scorer.score(&stats).as_f32();
    let expected = bm25_reference(&stats, Bm25Params::new());

    assert_close_f32(actual, expected, "bm25 known-answer score");
}

#[test]
fn test_bm25_zero_term_frequency() {
    let scorer = Bm25Scorer::new();
    let stats = ScoringStats {
        term_frequency: 0,
        doc_length: 100,
        avg_doc_length: 150.0,
        doc_count: 1000,
        doc_frequency: 50,
        ..ScoringStats::new()
    };

    assert_eq!(scorer.score(&stats), Score::ZERO);
}

#[test]
fn test_bm25_zero_average_document_length_is_safe() {
    let scorer = Bm25Scorer::new();
    let stats = ScoringStats {
        term_frequency: 3,
        doc_length: 100,
        avg_doc_length: 0.0,
        doc_count: 1000,
        doc_frequency: 50,
        ..ScoringStats::new()
    };

    let actual = scorer.score(&stats).as_f32();
    let expected = bm25_reference(&stats, Bm25Params::new());

    assert_close_f32(
        actual,
        0.0,
        "bm25 should return zero when avg_doc_length is zero",
    );
    assert!(
        expected.is_finite(),
        "reference bm25 should remain finite when avg_doc_length is zero"
    );
}

#[test]
fn test_bm25_custom_parameters() {
    let params = Bm25Params::new().with_k1(2.0).with_b(0.5);
    let scorer = Bm25Scorer::with_params(params);
    let stats = ScoringStats {
        term_frequency: 3,
        doc_length: 100,
        avg_doc_length: 150.0,
        doc_count: 1000,
        doc_frequency: 50,
        ..ScoringStats::new()
    };

    let actual = scorer.score(&stats).as_f32();
    let expected = bm25_reference(&stats, params);

    assert_close_f32(actual, expected, "bm25 custom parameter score");
}

#[test]
fn test_bm25f_known_answer() {
    let scorer = Bm25FScorer::new();
    let fields = [
        FieldStats {
            field_id: leit_core::FieldId::new(0),
            term_frequency: 2,
            field_length: 50,
            weight: 2.0,
        },
        FieldStats {
            field_id: leit_core::FieldId::new(1),
            term_frequency: 1,
            field_length: 100,
            weight: 1.0,
        },
    ];

    let actual = scorer.score(&fields, 150.0, 1000, 50).as_f32();
    let expected = bm25f_reference(&fields, 150.0, 1000, 50, Bm25Params::new());

    assert_close_f32(actual, expected, "bm25f known-answer score");
}

#[test]
fn test_bm25f_zero_weighted_tf() {
    let scorer = Bm25FScorer::new();
    let fields = [
        FieldStats {
            field_id: leit_core::FieldId::new(0),
            term_frequency: 0,
            field_length: 50,
            weight: 2.0,
        },
        FieldStats {
            field_id: leit_core::FieldId::new(1),
            term_frequency: 0,
            field_length: 100,
            weight: 1.0,
        },
    ];

    assert_eq!(scorer.score(&fields, 150.0, 1000, 50), Score::ZERO);
}

proptest! {
    #[test]
    fn test_lexical_scorers_match_reference_models(
        bm25_term_frequency in 1_u32..50_u32,
        bm25_doc_length in 1_u32..500_u32,
        bm25_avg_doc_length in 0.1_f32..1_000.0_f32,
        bm25_doc_count in 1_u32..10_000_u32,
        bm25_doc_frequency in 1_u32..10_000_u32,
        bm25f_fields in vec((1_u32..25_u32, 1_u32..500_u32, 0.1_f32..4.0_f32), 1..4),
        bm25f_avg_doc_length in 0.1_f32..1_000.0_f32,
        bm25f_doc_count in 1_u32..10_000_u32,
        bm25f_doc_frequency in 1_u32..10_000_u32,
    ) {
        prop_assume!(bm25_doc_frequency <= bm25_doc_count);
        prop_assume!(bm25f_doc_frequency <= bm25f_doc_count);

        let params = Bm25Params::new();

        let bm25_stats = ScoringStats {
            term_frequency: bm25_term_frequency,
            doc_length: bm25_doc_length,
            avg_doc_length: bm25_avg_doc_length,
            doc_count: bm25_doc_count,
            doc_frequency: bm25_doc_frequency,
            ..ScoringStats::new()
        };

        let bm25_actual = Bm25Scorer::with_params(params).score(&bm25_stats).as_f32();
        let bm25_expected = bm25_reference(&bm25_stats, params);
        prop_assert!(
            (bm25_actual - bm25_expected).abs() <= F32_EPSILON,
            "bm25 reference mismatch: expected {bm25_expected:.8}, got {bm25_actual:.8}"
        );

        let bm25f_fields: Vec<_> = bm25f_fields
            .into_iter()
            .enumerate()
            .map(|(index, (term_frequency, field_length, weight))| FieldStats {
                field_id: leit_core::FieldId::new(
                    u32::try_from(index).expect("test field index should fit in u32")
                ),
                term_frequency,
                field_length,
                weight,
            })
            .collect();

        let bm25f_actual = Bm25FScorer::new()
            .score(&bm25f_fields, bm25f_avg_doc_length, bm25f_doc_count, bm25f_doc_frequency)
            .as_f32();
        let bm25f_expected = bm25f_reference(
            &bm25f_fields,
            bm25f_avg_doc_length,
            bm25f_doc_count,
            bm25f_doc_frequency,
            params,
        );
        prop_assert!(
            (bm25f_actual - bm25f_expected).abs() <= F32_EPSILON,
            "bm25f reference mismatch: expected {bm25f_expected:.8}, got {bm25f_actual:.8}"
        );
    }
}

#[test]
fn test_topk_collector_basic() {
    let mut collector = TopKCollector::<u32>::new(3);
    collector.begin_query();

    collector.collect_scored(ScoredHit::new(1, Score::new(0.5)));
    collector.collect_scored(ScoredHit::new(2, Score::new(0.8)));
    collector.collect_scored(ScoredHit::new(3, Score::new(0.3)));
    collector.collect_scored(ScoredHit::new(4, Score::new(0.9)));
    collector.collect_scored(ScoredHit::new(5, Score::new(0.1)));

    let hits = collector.finish();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].id, 4);
    assert_eq!(hits[1].id, 2);
    assert_eq!(hits[2].id, 1);
}

#[test]
fn test_topk_collector_early_termination() {
    let mut collector = TopKCollector::<u32>::new(2);
    collector.begin_query();

    assert_eq!(collector.min_competitive_score(), None);

    collector.collect_scored(ScoredHit::new(1, Score::new(0.5)));
    collector.collect_scored(ScoredHit::new(2, Score::new(0.8)));

    assert_eq!(collector.min_competitive_score(), Some(Score::new(0.5)));
    assert!(collector.can_skip(Score::new(0.3)));
}

#[test]
fn test_count_collector_trait_contract() {
    let mut collector = CountCollector::new();
    <CountCollector as Collector<u32>>::begin_query(&mut collector);

    assert_eq!(collector.count(), 0);
    assert!(collector.is_empty());

    collector.collect_match(1_u32);
    collector.collect_match(2_u32);
    collector.collect_match(3_u32);

    assert_eq!(collector.count(), 3);
    assert!(!collector.is_empty());
    assert_eq!(collector.finish(), 3);
}

#[derive(Default)]
struct TestScratch {
    cleared: bool,
}

impl ScratchSpace for TestScratch {
    fn clear(&mut self) {
        self.cleared = true;
    }
}

fn clear_workspace(workspace: &mut impl Workspace) {
    workspace.clear();
}

#[derive(Debug, Default)]
struct TestFieldRegistry;

impl FieldRegistry for TestFieldRegistry {
    fn resolve_field(&self, field: &str) -> Option<leit_core::FieldId> {
        match field {
            "title" => Some(leit_core::FieldId::new(1)),
            "body" => Some(leit_core::FieldId::new(2)),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
struct TestDictionary;

impl TermDictionary for TestDictionary {
    fn resolve_term(&self, field: leit_core::FieldId, term: &str) -> Option<leit_core::TermId> {
        match (field.as_u32(), term) {
            (1, "rust") => Some(leit_core::TermId::new(10)),
            (2, "retrieval") => Some(leit_core::TermId::new(20)),
            _ => None,
        }
    }
}

#[test]
fn test_workspace_seam_uses_scratch_contract() {
    let mut workspace = TestScratch::default();

    clear_workspace(&mut workspace);

    assert!(workspace.cleared);
}

#[test]
fn test_rrf_fusion_baseline() {
    let list_a = vec![
        RankedResult::new("alpha", 1),
        RankedResult::new("beta", 2),
        RankedResult::new("gamma", 3),
    ];
    let list_b = vec![
        RankedResult::new("beta", 1),
        RankedResult::new("delta", 2),
        RankedResult::new("alpha", 3),
    ];

    let fused = fuse_default(&[list_a, list_b]);

    assert_eq!(fused.len(), 4);
    assert_eq!(fused[0].id, "beta");
    assert_eq!(fused[1].id, "alpha");
    assert_eq!(fused[0].rank, 1);
    assert_eq!(fused[1].rank, 2);

    let expected_beta = (1.0 / 62.0) + (1.0 / 61.0);
    let expected_alpha = (1.0 / 61.0) + (1.0 / 63.0);
    assert_close_f64(fused[0].score, expected_beta, "rrf score for beta");
    assert_close_f64(fused[1].score, expected_alpha, "rrf score for alpha");
}

#[test]
fn test_query_execution() {
    let planner = Planner::new();
    let dictionary = TestDictionary;
    let fields = TestFieldRegistry;
    let mut scratch = PlannerScratch::new();
    let context = PlanningContext::new(&dictionary, &fields)
        .with_default_field(leit_core::FieldId::new(2))
        .with_default_boost(1.0);

    let plan = planner
        .plan("title:rust OR retrieval", &context, &mut scratch)
        .expect("phase 1 query planning should succeed");

    assert_eq!(plan.program.node_count(), 3);
    assert_eq!(plan.program.max_depth(), 2);
    assert_eq!(plan.required_features, FeatureSet::basic());

    match plan.program.get(plan.program.root()) {
        Some(QueryNode::Or { children, boost }) => {
            assert_close_f32(*boost, 1.0, "default planning boost");
            assert_eq!(children.len(), 2);
        }
        other => panic!("expected OR root, got {other:?}"),
    }
}

#[test]
fn test_inverted_index_build() {
    let mut analyzers = FieldAnalyzers::new();
    analyzers.set(
        leit_core::FieldId::new(1),
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new()),
    );
    analyzers.set(
        leit_core::FieldId::new(2),
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new()),
    );

    let mut builder = InMemoryIndexBuilder::new(analyzers);
    builder
        .index_document(
            1,
            &[
                (leit_core::FieldId::new(1), "Rust Search"),
                (leit_core::FieldId::new(2), "Retrieval systems in rust"),
            ],
        )
        .expect("document 1 should index");
    builder
        .index_document(
            2,
            &[
                (leit_core::FieldId::new(1), "Search Engines"),
                (leit_core::FieldId::new(2), "Rust indexing pipeline"),
            ],
        )
        .expect("document 2 should index");
    let index = builder.build_index();

    let segment = index
        .to_segment_bytes()
        .expect("segment export should succeed");
    let view = SegmentView::open(&segment).expect("segment view should open");

    assert_eq!(view.document_count(), 2);
    assert_eq!(view.field_count(), 2);
    assert!(view.term_count() >= 4);
    assert!(view.has_section(SectionKind::TermDictionary));
    assert!(view.has_section(SectionKind::FieldMetadata));
    assert!(view.has_section(SectionKind::PostingsMetadata));
    assert!(view.has_section(SectionKind::PostingsPayload));
}

#[test]
fn test_text_analysis() {
    let analyzer =
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new());

    let analyzed_tokens = analyzer.analyze("Rust Retrieval");

    assert_eq!(analyzed_tokens.len(), 2);
    assert_eq!(analyzed_tokens[0].1, "rust");
    assert_eq!(analyzed_tokens[1].1, "retrieval");
}

#[test]
fn test_postings_lists() {
    let mut list = leit_postings::PostingsList::<u32>::new(leit_core::TermId::new(7));
    list.add(leit_postings::Posting {
        doc_id: 3,
        term_freq: 2,
        positions: Some(vec![1, 4]),
    });

    let mut postings = leit_postings::InMemoryPostings::new();
    postings.add(list);

    let mut cursor = postings
        .cursor(leit_core::TermId::new(7))
        .expect("term cursor should exist");

    assert_eq!(cursor.doc(), Some(3));
    assert_eq!(cursor.term_freq(), 2);
    assert!(!cursor.advance());
}

#[test]
fn test_e2e_search_pipeline() {
    let mut analyzers = FieldAnalyzers::new();
    let analyzer =
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new());
    analyzers.set(leit_core::FieldId::new(1), analyzer);
    analyzers.set(
        leit_core::FieldId::new(2),
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new()),
    );

    let mut builder = InMemoryIndexBuilder::new(analyzers);
    builder.register_field_alias(leit_core::FieldId::new(1), "title");
    builder.register_field_alias(leit_core::FieldId::new(2), "body");
    builder
        .index_document(
            1,
            &[
                (leit_core::FieldId::new(1), "Rust Programming"),
                (
                    leit_core::FieldId::new(2),
                    "Rust is a systems programming language",
                ),
            ],
        )
        .expect("document 1 should index");
    builder
        .index_document(
            2,
            &[
                (leit_core::FieldId::new(1), "Information Retrieval"),
                (
                    leit_core::FieldId::new(2),
                    "Search engines use inverted indices for retrieval",
                ),
            ],
        )
        .expect("document 2 should index");
    builder
        .index_document(
            3,
            &[
                (leit_core::FieldId::new(1), "Cooking Notes"),
                (leit_core::FieldId::new(2), "Recipes and ingredients"),
            ],
        )
        .expect("document 3 should index");
    let index = builder.build_index();

    let mut workspace = ExecutionWorkspace::new();
    let hits = workspace
        .search(
            &index,
            "title:rust OR body:retrieval",
            10,
            SearchScorer::bm25(),
            &NoFilter,
        )
        .expect("search should succeed");

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, 1);
    assert_eq!(hits[1].id, 2);
    assert!(
        hits[0].score > Score::ZERO,
        "doc 1 should have positive score"
    );
    assert!(
        hits[1].score > Score::ZERO,
        "doc 2 should have positive score"
    );
    assert!(
        hits[0].score > hits[1].score,
        "doc 1 (rust in title+body) should score higher than doc 2 (retrieval in body only)"
    );
}

#[test]
fn test_e2e_search_pipeline_bm25f() {
    let mut analyzers = FieldAnalyzers::new();
    analyzers.set(
        leit_core::FieldId::new(1),
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new()),
    );
    analyzers.set(
        leit_core::FieldId::new(2),
        Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(UnicodeNormalizer::new()),
    );

    let mut builder = InMemoryIndexBuilder::new(analyzers);
    builder.register_field_alias(leit_core::FieldId::new(1), "title");
    builder.register_field_alias(leit_core::FieldId::new(2), "body");
    builder
        .index_document(
            1,
            &[
                (leit_core::FieldId::new(1), "Rust Programming"),
                (
                    leit_core::FieldId::new(2),
                    "Rust is a systems programming language",
                ),
            ],
        )
        .expect("document 1 should index");
    builder
        .index_document(
            2,
            &[
                (leit_core::FieldId::new(1), "Information Retrieval"),
                (
                    leit_core::FieldId::new(2),
                    "Search engines use inverted indices for retrieval",
                ),
            ],
        )
        .expect("document 2 should index");
    builder
        .index_document(
            3,
            &[
                (leit_core::FieldId::new(1), "Cooking Notes"),
                (leit_core::FieldId::new(2), "Recipes and ingredients"),
            ],
        )
        .expect("document 3 should index");
    let index = builder.build_index();

    let mut workspace = ExecutionWorkspace::new();
    let hits = workspace
        .search(
            &index,
            "title:rust OR body:retrieval",
            10,
            SearchScorer::bm25f(),
            &NoFilter,
        )
        .expect("bm25f search should succeed");

    assert_eq!(hits.len(), 2, "bm25f should find 2 matching documents");
    assert_eq!(hits[0].id, 1);
    assert_eq!(hits[1].id, 2);
    assert!(
        hits[0].score > Score::ZERO,
        "bm25f doc 1 should have positive score"
    );
    assert!(
        hits[1].score > Score::ZERO,
        "bm25f doc 2 should have positive score"
    );
    assert!(
        hits[0].score > hits[1].score,
        "bm25f doc 1 should score higher than doc 2"
    );
}
