// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::{AddAssign, MulAssign};

use leit_collect::Collector;
use leit_core::{FieldId, FilterEvaluator, QueryNodeId, Score, ScoredHit, TermId};
use leit_query::{ExecutionPlan, FieldRegistry, QueryNode, QueryProgram, TermDictionary};
use leit_text::FieldAnalyzers;

use crate::codec::encode_segment;
use crate::error::IndexError;
use crate::search::{ExecutionStats, FieldHit, SearchScorer};

pub(crate) const DEFAULT_POSTINGS_BLOCK_SIZE: usize = 2;

#[derive(Clone, Debug)]
pub(crate) struct TermEntry {
    pub(crate) field_id: FieldId,
    pub(crate) term_id: TermId,
    pub(crate) term: String,
}

/// A single posting: a document ID and its term frequency for a term.
///
/// Postings are aggregated per term and stored in doc-sorted order (ascending doc ID).
/// This type is public primarily for codec benchmarking and analysis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PostingEntry {
    /// Document identifier (segment-local, u32).
    pub doc_id: u32,
    /// Term frequency (raw count of term occurrences in the document's field).
    pub term_freq: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FieldMetadata {
    pub(crate) field_id: FieldId,
    pub(crate) doc_count: u32,
    pub(crate) total_terms: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PostingBlock {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) end_doc: u32,
    pub(crate) max_term_freq: u32,
    pub(crate) min_doc_length: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct EvalResult {
    pub(crate) matches: BTreeSet<u32>,
    pub(crate) scores: BTreeMap<u32, Score>,
}

impl EvalResult {
    pub(crate) fn from_scores(scores: BTreeMap<u32, Score>) -> Self {
        let matches = scores.keys().copied().collect();
        Self { matches, scores }
    }

    const fn from_matches(matches: BTreeSet<u32>) -> Self {
        Self {
            matches,
            scores: BTreeMap::new(),
        }
    }
}

fn is_non_unit_boost(boost: f32) -> bool {
    debug_assert!(boost.is_finite(), "boost must be finite");
    (boost - 1.0).abs() > f32::EPSILON
}

const fn u32_to_f32(value: u32) -> f32 {
    value as f32
}

/// An immutable searchable in-memory Phase 1 index.
#[derive(Debug)]
pub struct InMemoryIndex {
    pub(crate) analyzers: FieldAnalyzers,
    pub(crate) documents: BTreeSet<u32>,
    pub(crate) terms_to_ids: BTreeMap<(FieldId, String), TermId>,
    pub(crate) term_entries: Vec<TermEntry>,
    pub(crate) postings: BTreeMap<TermId, Vec<PostingEntry>>,
    pub(crate) posting_blocks: BTreeMap<TermId, Vec<PostingBlock>>,
    pub(crate) field_stats: BTreeMap<FieldId, FieldMetadata>,
    pub(crate) field_names: BTreeMap<String, FieldId>,
    pub(crate) field_doc_lengths: BTreeMap<(u32, FieldId), u32>,
}

impl InMemoryIndex {
    pub(crate) fn new(
        analyzers: FieldAnalyzers,
        documents: BTreeSet<u32>,
        terms_to_ids: BTreeMap<(FieldId, String), TermId>,
        term_entries: Vec<TermEntry>,
        postings: BTreeMap<TermId, Vec<PostingEntry>>,
        posting_blocks: BTreeMap<TermId, Vec<PostingBlock>>,
        field_stats: BTreeMap<FieldId, FieldMetadata>,
        field_names: BTreeMap<String, FieldId>,
        field_doc_lengths: BTreeMap<(u32, FieldId), u32>,
    ) -> Self {
        Self {
            analyzers,
            documents,
            terms_to_ids,
            term_entries,
            postings,
            posting_blocks,
            field_stats,
            field_names,
            field_doc_lengths,
        }
    }

    /// Serialize the current index into a single validated segment buffer.
    pub fn to_segment_bytes(&self) -> Result<Vec<u8>, IndexError> {
        encode_segment(self)
    }

    pub(crate) fn document_count(&self) -> u32 {
        u32::try_from(self.documents.len()).unwrap_or(u32::MAX)
    }

    pub(crate) fn term_entries(&self) -> &[TermEntry] {
        &self.term_entries
    }

    pub(crate) const fn field_stats(&self) -> &BTreeMap<FieldId, FieldMetadata> {
        &self.field_stats
    }

    pub(crate) const fn postings(&self) -> &BTreeMap<TermId, Vec<PostingEntry>> {
        &self.postings
    }

    /// Return all postings indexed by term ID.
    ///
    /// Postings are doc-sorted (ascending doc ID) within each term's list.
    /// This method is primarily used for codec benchmarking and analysis.
    pub fn postings_by_term(&self) -> &BTreeMap<TermId, Vec<PostingEntry>> {
        &self.postings
    }

    fn avg_field_doc_length(&self, field: FieldId) -> f32 {
        let Some(stats) = self.field_stats.get(&field) else {
            return 0.0;
        };
        if stats.doc_count == 0 {
            return 0.0;
        }
        u32_to_f32(stats.total_terms) / u32_to_f32(stats.doc_count)
    }

    pub(crate) fn default_fields(&self) -> Vec<FieldId> {
        let fields: Vec<FieldId> = self.field_stats.values().map(|s| s.field_id).collect();
        if fields.is_empty() {
            self.field_names.values().copied().collect()
        } else {
            fields
        }
    }

    pub(crate) fn evaluate_plan<F: FilterEvaluator<u32>>(
        &self,
        plan: &ExecutionPlan,
        scorer: SearchScorer,
        filter: &F,
        stats: &mut ExecutionStats,
    ) -> Result<EvalResult, IndexError> {
        self.evaluate_node(plan.program.root(), &plan.program, scorer, filter, stats)
    }

    pub(crate) fn evaluate_matches<F: FilterEvaluator<u32>>(
        &self,
        plan: &ExecutionPlan,
        filter: &F,
    ) -> Result<BTreeSet<u32>, IndexError> {
        self.evaluate_matches_node(plan.program.root(), &plan.program, filter)
    }

    fn evaluate_node<F: FilterEvaluator<u32>>(
        &self,
        node_id: QueryNodeId,
        program: &QueryProgram,
        scoring: SearchScorer,
        filter: &F,
        stats: &mut ExecutionStats,
    ) -> Result<EvalResult, IndexError> {
        let Some(node) = program.get(node_id) else {
            return Ok(EvalResult::default());
        };

        match node {
            QueryNode::Term { field, term, boost } => {
                Ok(self.eval_term(*field, *term, *boost, scoring, stats))
            }
            QueryNode::TermExpansion {
                children,
                fields,
                boost,
                field_weights,
            } => {
                if let SearchScorer::Bm25F(_) = scoring
                    && let Some(mut result) = self.eval_bm25f_term_expansion(
                        children,
                        fields,
                        field_weights,
                        program,
                        scoring,
                        stats,
                    )
                {
                    if is_non_unit_boost(*boost) {
                        for score in result.scores.values_mut() {
                            MulAssign::mul_assign(score, *boost);
                        }
                    }
                    return Ok(result);
                }

                self.eval_disjunction(children, *boost, program, scoring, filter, stats)
            }
            QueryNode::Or { children, boost } => {
                self.eval_disjunction(children, *boost, program, scoring, filter, stats)
            }
            QueryNode::And { children, boost } => {
                let mut iter = children.iter();
                let Some(first) = iter.next() else {
                    return Ok(EvalResult::default());
                };
                let first_result = self.evaluate_node(*first, program, scoring, filter, stats)?;
                let mut matches = first_result.matches.clone();
                let mut child_results = Vec::new();
                child_results.push(first_result);
                for child in iter {
                    let child_result =
                        self.evaluate_node(*child, program, scoring, filter, stats)?;
                    matches.retain(|doc_id| child_result.matches.contains(doc_id));
                    child_results.push(child_result);
                }
                let mut results = BTreeMap::new();
                for child_result in child_results {
                    for (doc_id, child_score) in child_result.scores {
                        if matches.contains(&doc_id) {
                            let entry = results.entry(doc_id).or_insert(Score::ZERO);
                            AddAssign::add_assign(entry, child_score);
                        }
                    }
                }
                if is_non_unit_boost(*boost) {
                    for score in results.values_mut() {
                        MulAssign::mul_assign(score, *boost);
                    }
                }
                Ok(EvalResult {
                    matches,
                    scores: results,
                })
            }
            QueryNode::Not { child } => {
                let child_matches = self
                    .evaluate_node(*child, program, scoring, filter, stats)?
                    .matches;
                let mut matches = BTreeSet::new();
                for doc_id in &self.documents {
                    if !child_matches.contains(doc_id) {
                        matches.insert(*doc_id);
                    }
                }
                Ok(EvalResult::from_matches(matches))
            }
            QueryNode::ConstantScore { child, score } => {
                let mut result = self.evaluate_node(*child, program, scoring, filter, stats)?;
                result.scores.clear();
                let safe_score = Score::try_from(*score).unwrap_or(Score::ZERO);
                for doc_id in &result.matches {
                    result.scores.insert(*doc_id, safe_score);
                }
                Ok(result)
            }
            QueryNode::ExternalFilter { input, slot } => {
                let mut result = self.evaluate_node(*input, program, scoring, filter, stats)?;
                result
                    .matches
                    .retain(|doc_id| filter.evaluate(*slot, doc_id));
                result
                    .scores
                    .retain(|doc_id, _| result.matches.contains(doc_id));
                Ok(result)
            }
            QueryNode::Filter { .. } => Err(IndexError::UnsupportedFilterPredicate),
        }
    }

    fn eval_disjunction<F: FilterEvaluator<u32>>(
        &self,
        children: &[QueryNodeId],
        boost: f32,
        program: &QueryProgram,
        scoring: SearchScorer,
        filter: &F,
        stats: &mut ExecutionStats,
    ) -> Result<EvalResult, IndexError> {
        let mut matches = BTreeSet::new();
        let mut results = BTreeMap::new();
        for child in children {
            let child_result = self.evaluate_node(*child, program, scoring, filter, stats)?;
            matches.extend(child_result.matches);
            for (doc_id, score) in child_result.scores {
                let entry = results.entry(doc_id).or_insert(Score::ZERO);
                AddAssign::add_assign(entry, score);
            }
        }
        if is_non_unit_boost(boost) {
            for score in results.values_mut() {
                MulAssign::mul_assign(score, boost);
            }
        }
        Ok(EvalResult {
            matches,
            scores: results,
        })
    }

    fn eval_bm25f_term_expansion(
        &self,
        children: &[QueryNodeId],
        fields: &[FieldId],
        field_weights: &BTreeMap<FieldId, f32>,
        program: &QueryProgram,
        scoring: SearchScorer,
        stats: &mut ExecutionStats,
    ) -> Option<EvalResult> {
        let mut terms = Vec::with_capacity(children.len());
        let mut seen_fields = BTreeSet::new();
        let mut expected_text: Option<&str> = None;
        let mut expected_boost: Option<f32> = None;
        for child in children {
            let QueryNode::Term { field, term, boost } = program.get(*child)? else {
                return None;
            };
            if !seen_fields.insert(*field) {
                return None;
            }
            let term_entry = self.term_entries.get(term.as_u32() as usize)?;
            if term_entry.term_id != *term || term_entry.field_id != *field {
                return None;
            }
            match expected_text {
                Some(text) if text != term_entry.term.as_str() => return None,
                Some(_) => {}
                None => expected_text = Some(term_entry.term.as_str()),
            }
            match expected_boost {
                Some(value) if (value - *boost).abs() > f32::EPSILON => return None,
                Some(_) => {}
                None => expected_boost = Some(*boost),
            }
            terms.push((*field, *term));
        }

        let weight = |field: FieldId| -> f32 { field_weights.get(&field).copied().unwrap_or(1.0) };

        let mut aggregation_fields = Vec::with_capacity(fields.len());
        let mut avg_doc_length = 0.0_f32;
        for &field in fields {
            let avg_field_length = self.avg_field_doc_length(field);
            avg_doc_length += avg_field_length;
            aggregation_fields.push((field, avg_field_length));
        }

        let mut hits_by_doc = BTreeMap::<u32, BTreeMap<FieldId, FieldHit>>::new();
        for (field, term) in terms {
            let postings = self.postings.get(&term)?;
            let avg_field_length = self.avg_field_doc_length(field);
            for posting in postings {
                stats.scored_postings = stats.scored_postings.saturating_add(1);
                let field_length = self
                    .field_doc_lengths
                    .get(&(posting.doc_id, field))
                    .copied()
                    .unwrap_or_default();
                hits_by_doc.entry(posting.doc_id).or_default().insert(
                    field,
                    FieldHit {
                        field,
                        term_frequency: posting.term_freq,
                        field_length,
                        avg_field_length,
                        weight: weight(field),
                    },
                );
            }
        }

        let doc_count = self.document_count();
        let doc_frequency = u32::try_from(hits_by_doc.len()).unwrap_or(u32::MAX);
        let boost = expected_boost.unwrap_or(1.0);
        let mut scores = BTreeMap::new();
        for (doc_id, mut field_hits_by_field) in hits_by_doc {
            for (field, avg_field_length) in &aggregation_fields {
                field_hits_by_field
                    .entry(*field)
                    .or_insert_with(|| FieldHit {
                        field: *field,
                        term_frequency: 0,
                        field_length: self
                            .field_doc_lengths
                            .get(&(doc_id, *field))
                            .copied()
                            .unwrap_or_default(),
                        avg_field_length: *avg_field_length,
                        weight: weight(*field),
                    });
            }
            let field_hits: Vec<FieldHit> = field_hits_by_field.into_values().collect();
            let score = scoring.score_term_fields(
                &field_hits,
                avg_doc_length,
                doc_count,
                doc_frequency,
                boost,
            );
            scores.insert(doc_id, score);
        }
        Some(EvalResult::from_scores(scores))
    }

    fn evaluate_matches_node<F: FilterEvaluator<u32>>(
        &self,
        node_id: QueryNodeId,
        program: &QueryProgram,
        filter: &F,
    ) -> Result<BTreeSet<u32>, IndexError> {
        let Some(node) = program.get(node_id) else {
            return Ok(BTreeSet::new());
        };

        match node {
            QueryNode::Term { term, .. } => {
                let mut matches = BTreeSet::new();
                if let Some(postings) = self.postings.get(term) {
                    for posting in postings {
                        matches.insert(posting.doc_id);
                    }
                }
                Ok(matches)
            }
            QueryNode::Or { children, .. } | QueryNode::TermExpansion { children, .. } => {
                let mut matches = BTreeSet::new();
                for child in children {
                    matches.extend(self.evaluate_matches_node(*child, program, filter)?);
                }
                Ok(matches)
            }
            QueryNode::And { children, .. } => {
                let mut iter = children.iter();
                let Some(first) = iter.next() else {
                    return Ok(BTreeSet::new());
                };
                let mut matches = self.evaluate_matches_node(*first, program, filter)?;
                for child in iter {
                    let child_matches = self.evaluate_matches_node(*child, program, filter)?;
                    matches.retain(|doc_id| child_matches.contains(doc_id));
                }
                Ok(matches)
            }
            QueryNode::Not { child } => {
                let child_matches = self.evaluate_matches_node(*child, program, filter)?;
                let mut matches = BTreeSet::new();
                for doc_id in &self.documents {
                    if !child_matches.contains(doc_id) {
                        matches.insert(*doc_id);
                    }
                }
                Ok(matches)
            }
            QueryNode::ConstantScore { child, .. } => {
                self.evaluate_matches_node(*child, program, filter)
            }
            QueryNode::ExternalFilter { input, slot } => {
                let mut matches = self.evaluate_matches_node(*input, program, filter)?;
                matches.retain(|doc_id| filter.evaluate(*slot, doc_id));
                Ok(matches)
            }
            QueryNode::Filter { .. } => Err(IndexError::UnsupportedFilterPredicate),
        }
    }

    fn score_posting(
        &self,
        posting: &PostingEntry,
        field: FieldId,
        boost: f32,
        scoring: SearchScorer,
        avg_doc_length: f32,
        doc_count: u32,
        doc_frequency: u32,
    ) -> Score {
        let doc_length = self
            .field_doc_lengths
            .get(&(posting.doc_id, field))
            .copied()
            .unwrap_or_default();
        let mut score = scoring.score_term(
            field,
            posting.term_freq,
            doc_length,
            avg_doc_length,
            doc_count,
            doc_frequency,
        );
        if is_non_unit_boost(boost) {
            MulAssign::mul_assign(&mut score, boost);
        }
        score
    }

    fn eval_term(
        &self,
        field: FieldId,
        term: TermId,
        boost: f32,
        scoring: SearchScorer,
        stats: &mut ExecutionStats,
    ) -> EvalResult {
        let mut results = BTreeMap::new();
        let Some(postings) = self.postings.get(&term) else {
            return EvalResult::default();
        };

        let avg_doc_length = self.avg_field_doc_length(field);
        let doc_count = self.document_count();
        let doc_frequency = u32::try_from(postings.len()).unwrap_or(u32::MAX);

        for posting in postings {
            stats.scored_postings = stats.scored_postings.saturating_add(1);
            let score = self.score_posting(
                posting,
                field,
                boost,
                scoring,
                avg_doc_length,
                doc_count,
                doc_frequency,
            );
            results.insert(posting.doc_id, score);
        }

        EvalResult::from_scores(results)
    }

    pub(crate) fn collect_result<S>(
        result: EvalResult,
        collectors: &mut S,
        stats: &mut ExecutionStats,
        allow_pruning: bool,
    ) where
        S: Collector<u32> + ?Sized,
    {
        for doc_id in result.matches {
            let score = result.scores.get(&doc_id).copied().unwrap_or(Score::ZERO);
            if allow_pruning && collectors.can_skip(score) {
                continue;
            }
            collectors.collect_scored(ScoredHit::new(doc_id, score));
            stats.collected_hits = stats.collected_hits.saturating_add(1);
        }
    }

    pub(crate) fn collect_matches<S>(
        matches: BTreeSet<u32>,
        collectors: &mut S,
        stats: &mut ExecutionStats,
    ) where
        S: Collector<u32> + ?Sized,
    {
        for doc_id in matches {
            collectors.collect_match(doc_id);
            stats.collected_hits = stats.collected_hits.saturating_add(1);
        }
    }

    /// Try to execute the plan root via an optimized fast path.
    ///
    /// Returns `Ok(true)` if handled, `Ok(false)` to fall through to the
    /// general evaluator. The `filter` parameter is threaded to recursive
    /// calls (e.g. `ConstantScore` → `evaluate_node`) but is not consulted
    /// on leaf fast paths (`Term`) because those only fire when the root is
    /// a bare `Term` node with no `ExternalFilter` wrapping. Filter dispatch
    /// is node-mediated via `ExternalFilter` nodes in the general evaluator.
    pub(crate) fn try_execute_root<S, F>(
        &self,
        plan: &ExecutionPlan,
        scoring: SearchScorer,
        collectors: &mut S,
        stats: &mut ExecutionStats,
        allow_pruning: bool,
        filter: &F,
    ) -> Result<bool, IndexError>
    where
        S: Collector<u32> + ?Sized,
        F: FilterEvaluator<u32>,
    {
        let Some(node) = plan.program.get(plan.program.root()) else {
            return Ok(true);
        };
        match node {
            QueryNode::Term { field, term, boost } => {
                debug_assert!(
                    filter.slots().is_empty(),
                    "Term fast path fired with active filter slots; \
                     ensure plan() was called with the same filter as execute()"
                );
                self.collect_term(
                    *field,
                    *term,
                    *boost,
                    scoring,
                    collectors,
                    stats,
                    allow_pruning,
                );
                Ok(true)
            }
            QueryNode::TermExpansion {
                children, boost, ..
            } if matches!(scoring, SearchScorer::Bm25(_)) && children.len() == 1 => {
                let Some(QueryNode::Term {
                    field,
                    term,
                    boost: term_boost,
                }) = plan.program.get(children[0])
                else {
                    return Ok(false);
                };
                debug_assert!(
                    filter.slots().is_empty(),
                    "TermExpansion fast path fired with active filter slots; \
                     ensure plan() was called with the same filter as execute()"
                );
                self.collect_term(
                    *field,
                    *term,
                    *term_boost * *boost,
                    scoring,
                    collectors,
                    stats,
                    allow_pruning,
                );
                Ok(true)
            }
            QueryNode::ConstantScore { child, score } => {
                let mut result =
                    self.evaluate_node(*child, &plan.program, scoring, filter, stats)?;
                result.scores.clear();
                let score = Score::try_from(*score).unwrap_or(Score::ZERO);
                if allow_pruning && collectors.can_skip(score) {
                    return Ok(true);
                }
                for doc_id in result.matches {
                    collectors.collect_scored(ScoredHit::new(doc_id, score));
                    stats.collected_hits = stats.collected_hits.saturating_add(1);
                }
                Ok(true)
            }
            QueryNode::Filter { .. } | QueryNode::ExternalFilter { .. } => Ok(false),
            _ => Ok(false),
        }
    }

    /// Unscored variant of [`try_execute_root`](Self::try_execute_root).
    ///
    /// Same fast-path semantics: `filter` is threaded to recursive calls but
    /// not consulted on the bare `Term` leaf path.
    pub(crate) fn try_execute_root_unscored<S, F>(
        &self,
        plan: &ExecutionPlan,
        collectors: &mut S,
        stats: &mut ExecutionStats,
        filter: &F,
    ) -> Result<bool, IndexError>
    where
        S: Collector<u32> + ?Sized,
        F: FilterEvaluator<u32>,
    {
        let Some(node) = plan.program.get(plan.program.root()) else {
            return Ok(true);
        };
        match node {
            QueryNode::Term { term, .. } => {
                debug_assert!(
                    filter.slots().is_empty(),
                    "Term fast path fired with active filter slots; \
                     ensure plan() was called with the same filter as execute()"
                );
                self.collect_term_docs(*term, collectors, stats);
                Ok(true)
            }
            QueryNode::ConstantScore { child, .. } => {
                let matches = self.evaluate_matches_node(*child, &plan.program, filter)?;
                Self::collect_matches(matches, collectors, stats);
                Ok(true)
            }
            QueryNode::Filter { .. } | QueryNode::ExternalFilter { .. } => Ok(false),
            _ => Ok(false),
        }
    }

    fn collect_term<S>(
        &self,
        field: FieldId,
        term: TermId,
        boost: f32,
        scoring: SearchScorer,
        collectors: &mut S,
        stats: &mut ExecutionStats,
        allow_pruning: bool,
    ) where
        S: Collector<u32> + ?Sized,
    {
        let Some(postings) = self.postings.get(&term) else {
            return;
        };
        let Some(blocks) = self.posting_blocks.get(&term) else {
            return;
        };

        let avg_doc_length = self.avg_field_doc_length(field);
        let doc_count = self.document_count();
        let doc_frequency = u32::try_from(postings.len()).unwrap_or(u32::MAX);

        for block in blocks {
            // Block-max pruning is only valid for non-negative boosts.
            // Negative boost inverts the upper bound, making it a lower bound.
            if allow_pruning
                && boost >= 0.0
                && let Some(threshold) = collectors.min_competitive_score()
            {
                let bound = Self::block_upper_bound(
                    *block,
                    field,
                    boost,
                    scoring,
                    avg_doc_length,
                    doc_count,
                    doc_frequency,
                );
                if bound < threshold {
                    stats.skipped_blocks = stats.skipped_blocks.saturating_add(1);
                    continue;
                }
            }

            for posting in &postings[block.start..block.end] {
                stats.scored_postings = stats.scored_postings.saturating_add(1);
                let score = self.score_posting(
                    posting,
                    field,
                    boost,
                    scoring,
                    avg_doc_length,
                    doc_count,
                    doc_frequency,
                );
                if allow_pruning && collectors.can_skip(score) {
                    continue;
                }
                collectors.collect_scored(ScoredHit::new(posting.doc_id, score));
                stats.collected_hits = stats.collected_hits.saturating_add(1);
            }
        }
    }

    fn collect_term_docs<S>(&self, term: TermId, collectors: &mut S, stats: &mut ExecutionStats)
    where
        S: Collector<u32> + ?Sized,
    {
        let Some(postings) = self.postings.get(&term) else {
            return;
        };
        for posting in postings {
            collectors.collect_match(posting.doc_id);
            stats.collected_hits = stats.collected_hits.saturating_add(1);
        }
    }

    fn block_upper_bound(
        block: PostingBlock,
        field: FieldId,
        boost: f32,
        scoring: SearchScorer,
        avg_doc_length: f32,
        doc_count: u32,
        doc_frequency: u32,
    ) -> Score {
        let mut bound = scoring.score_term(
            field,
            block.max_term_freq,
            block.min_doc_length,
            avg_doc_length,
            doc_count,
            doc_frequency,
        );
        if is_non_unit_boost(boost) {
            MulAssign::mul_assign(&mut bound, boost);
        }
        bound
    }
}

impl FieldRegistry for InMemoryIndex {
    fn resolve_field(&self, field: &str) -> Option<FieldId> {
        self.field_names.get(field).copied()
    }
}

impl TermDictionary for InMemoryIndex {
    fn resolve_term(&self, field: FieldId, term: &str) -> Option<TermId> {
        let analyzer = self.analyzers.get(field)?;
        let analyzed_tokens = analyzer.analyze(term);
        if analyzed_tokens.len() != 1 {
            return None;
        }
        let normalized = analyzed_tokens[0].1.as_str();
        self.terms_to_ids.get(&(field, normalized.into())).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use crate::builder::build_posting_blocks;

    #[test]
    fn posting_blocks_respect_configured_block_size() {
        let term_id = TermId::new(0);
        let term_entries = vec![TermEntry {
            field_id: FieldId::new(1),
            term_id,
            term: String::from("alpha"),
        }];
        let postings = BTreeMap::from([(
            term_id,
            vec![
                PostingEntry {
                    doc_id: 1,
                    term_freq: 3,
                },
                PostingEntry {
                    doc_id: 2,
                    term_freq: 2,
                },
                PostingEntry {
                    doc_id: 3,
                    term_freq: 1,
                },
            ],
        )]);
        let field_doc_lengths = BTreeMap::from([
            ((1, FieldId::new(1)), 5),
            ((2, FieldId::new(1)), 7),
            ((3, FieldId::new(1)), 9),
        ]);

        let singleton_blocks =
            build_posting_blocks(&term_entries, &postings, &field_doc_lengths, 1);
        let pair_blocks = build_posting_blocks(&term_entries, &postings, &field_doc_lengths, 2);

        assert_eq!(singleton_blocks[&term_id].len(), 3);
        assert_eq!(pair_blocks[&term_id].len(), 2);
        assert_eq!(
            pair_blocks[&term_id][0],
            PostingBlock {
                start: 0,
                end: 2,
                end_doc: 2,
                max_term_freq: 3,
                min_doc_length: 5,
            }
        );
    }
}
