// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Contract tests for planning typed `UserQueryProgram` ASTs.
//!
//! `Planner::plan_program` must mirror the textual `plan` lowering semantics:
//! same field resolution, default-field expansion, boost composition, and the
//! same `max_depth` / `max_nodes` guards.

use leit_core::{FieldId, QueryNodeId, TermId};
use leit_query::{
    FieldRegistry, Planner, PlannerScratch, PlanningContext, QueryBuilder, QueryError, QueryNode,
    TermDictionary, UserQueryProgram,
};

#[derive(Debug, Default)]
struct TestFieldRegistry;

impl FieldRegistry for TestFieldRegistry {
    fn resolve_field(&self, field: &str) -> Option<FieldId> {
        match field {
            "title" => Some(Self::title()),
            "body" => Some(Self::body()),
            _ => None,
        }
    }
}

impl TestFieldRegistry {
    const fn title() -> FieldId {
        FieldId::new(1)
    }

    const fn body() -> FieldId {
        FieldId::new(2)
    }
}

#[derive(Debug, Default)]
struct TestDictionary;

impl TermDictionary for TestDictionary {
    fn resolve_term(&self, field: FieldId, term: &str) -> Option<TermId> {
        match (field, term) {
            (field, "rust")
                if field == TestFieldRegistry::title() || field == TestFieldRegistry::body() =>
            {
                Some(TermId::new(10))
            }
            (field, "memory") if field == TestFieldRegistry::body() => Some(TermId::new(20)),
            (field, "safety") if field == TestFieldRegistry::body() => Some(TermId::new(30)),
            _ => None,
        }
    }
}

fn context<'a>(
    dictionary: &'a TestDictionary,
    fields: &'a TestFieldRegistry,
) -> PlanningContext<'a> {
    PlanningContext::new(dictionary, fields).with_default_field(TestFieldRegistry::body())
}

fn assert_f32_eq(actual: f32, expected: f32) {
    let delta = (actual - expected).abs();
    assert!(delta <= f32::EPSILON, "expected {expected}, got {actual}");
}

fn plan_it(program: &UserQueryProgram) -> Result<leit_query::ExecutionPlan, QueryError> {
    let planner = Planner::new();
    let dictionary = TestDictionary;
    let fields = TestFieldRegistry;
    let mut scratch = PlannerScratch::new();
    planner.plan_program(program, &context(&dictionary, &fields), &mut scratch)
}

#[test]
fn term_lowers_to_default_field_term() {
    let program = leit_query::term("rust");
    let plan = plan_it(&program).expect("plan term");

    assert_eq!(plan.program.node_count(), 1);
    match plan.program.get(plan.program.root()) {
        Some(QueryNode::Term { field, term, boost }) => {
            assert_eq!(*field, TestFieldRegistry::body());
            assert_eq!(*term, TermId::new(10));
            assert_f32_eq(*boost, 1.0);
        }
        other => panic!("expected term node, got {other:?}"),
    }
}

#[test]
fn term_with_field_resolves_explicit_field() {
    let program = leit_query::term_with_field("rust", "title");
    let plan = plan_it(&program).expect("plan fielded term");

    match plan.program.get(plan.program.root()) {
        Some(QueryNode::Term { field, term, .. }) => {
            assert_eq!(*field, TestFieldRegistry::title());
            assert_eq!(*term, TermId::new(10));
        }
        other => panic!("expected term node, got {other:?}"),
    }
}

#[test]
fn unknown_field_is_rejected() {
    let program = leit_query::term_with_field("rust", "summary");
    let err = plan_it(&program).expect_err("unknown field must fail");
    assert!(matches!(err, QueryError::UnknownField { .. }));
}

#[test]
fn unresolved_term_lowers_to_match_nothing() {
    // Mirrors the textual planner: an out-of-dictionary term becomes an
    // empty AND (matches nothing) rather than an error.
    let program = leit_query::term("nonexistent");
    let plan = plan_it(&program).expect("plan unresolved term");
    match plan.program.get(plan.program.root()) {
        Some(QueryNode::And { children, .. }) => assert!(children.is_empty()),
        other => panic!("expected empty AND node, got {other:?}"),
    }
}

#[test]
fn phrase_lowers_to_conjunction_of_terms() {
    // Phase 1 execution has no positional data, so a phrase lowers to the
    // conjunction of its terms (slop is not enforceable yet).
    let program = leit_query::phrase(&["rust", "memory"]);
    let plan = plan_it(&program).expect("plan phrase");

    match plan.program.get(plan.program.root()) {
        Some(QueryNode::And { children, .. }) => {
            assert_eq!(children.len(), 2);
            for child in children {
                assert!(matches!(
                    plan.program.get(*child),
                    Some(QueryNode::Term { .. })
                ));
            }
        }
        other => panic!("expected AND node, got {other:?}"),
    }
}

#[test]
fn single_term_phrase_lowers_to_plain_term() {
    let program = leit_query::phrase(&["rust"]);
    let plan = plan_it(&program).expect("plan single-term phrase");
    assert_eq!(plan.program.node_count(), 1);
    assert!(matches!(
        plan.program.get(plan.program.root()),
        Some(QueryNode::Term { .. })
    ));
}

#[test]
fn boolean_nodes_lower_to_boolean_programs() {
    let mut builder = QueryBuilder::new();
    let a = builder.term("rust");
    let b = builder.term("memory");
    let or_node = builder.or(vec![a, b]);
    let c = builder.term("safety");
    let not_c = builder.not(c);
    builder.and(vec![or_node, not_c]);
    let program = builder.build().expect("build boolean program");

    let plan = plan_it(&program).expect("plan boolean program");

    match plan.program.get(plan.program.root()) {
        Some(QueryNode::And { children, .. }) => {
            assert_eq!(children.len(), 2);
            assert!(matches!(
                plan.program.get(children[0]),
                Some(QueryNode::Or { .. })
            ));
            assert!(matches!(
                plan.program.get(children[1]),
                Some(QueryNode::Not { .. })
            ));
        }
        other => panic!("expected AND root, got {other:?}"),
    }
}

#[test]
fn boost_multiplies_into_term_boost() {
    let mut builder = QueryBuilder::new();
    let t = builder.term("rust");
    builder.boost(t, 2.5);
    let program = builder.build().expect("build boosted program");

    let plan = plan_it(&program).expect("plan boosted term");
    match plan.program.get(plan.program.root()) {
        Some(QueryNode::Term { boost, .. }) => assert_f32_eq(*boost, 2.5),
        other => panic!("expected term node, got {other:?}"),
    }
}

#[test]
fn nested_boosts_compose_multiplicatively() {
    let mut builder = QueryBuilder::new();
    let t = builder.term("rust");
    let inner = builder.boost(t, 2.0);
    builder.boost(inner, 3.0);
    let program = builder.build().expect("build nested boost");

    let plan = plan_it(&program).expect("plan nested boost");
    match plan.program.get(plan.program.root()) {
        Some(QueryNode::Term { boost, .. }) => assert_f32_eq(*boost, 6.0),
        other => panic!("expected term node, got {other:?}"),
    }
}

#[test]
fn max_depth_guard_is_enforced() {
    let mut builder = QueryBuilder::new();
    let mut current = builder.term("rust");
    for _ in 0..4 {
        current = builder.not(current);
    }
    let program = builder.build().expect("build deep program");

    let planner = Planner::new().with_max_depth(3);
    let dictionary = TestDictionary;
    let fields = TestFieldRegistry;
    let mut scratch = PlannerScratch::new();
    let err = planner
        .plan_program(&program, &context(&dictionary, &fields), &mut scratch)
        .expect_err("depth guard must trip");
    assert!(matches!(err, QueryError::MaxDepthExceeded { .. }));
}

#[test]
fn max_nodes_guard_is_enforced() {
    let mut builder = QueryBuilder::new();
    let ids: Vec<QueryNodeId> = (0..8).map(|_| builder.term("rust")).collect();
    builder.or(ids);
    let program = builder.build().expect("build wide program");

    let planner = Planner::new().with_max_nodes(4);
    let dictionary = TestDictionary;
    let fields = TestFieldRegistry;
    let mut scratch = PlannerScratch::new();
    let err = planner
        .plan_program(&program, &context(&dictionary, &fields), &mut scratch)
        .expect_err("node guard must trip");
    assert!(matches!(err, QueryError::MaxNodesExceeded { .. }));
}

#[test]
fn typed_and_textual_plans_agree_for_equivalent_queries() {
    // The typed pipeline must produce the same execution plan as the textual
    // parser for a query expressible in both.
    let planner = Planner::new();
    let dictionary = TestDictionary;
    let fields = TestFieldRegistry;
    let ctx = context(&dictionary, &fields);
    let mut scratch = PlannerScratch::new();

    let textual = planner
        .plan("rust AND memory", &ctx, &mut scratch)
        .expect("textual plan");

    let mut builder = QueryBuilder::new();
    let a = builder.term("rust");
    let b = builder.term("memory");
    builder.and(vec![a, b]);
    let program = builder.build().expect("build program");

    scratch.reset();
    let typed = planner
        .plan_program(&program, &ctx, &mut scratch)
        .expect("typed plan");

    assert_eq!(textual, typed);
}

#[test]
fn multiple_default_fields_expand_terms() {
    let planner = Planner::new();
    let dictionary = TestDictionary;
    let fields = TestFieldRegistry;
    let ctx = PlanningContext::new(&dictionary, &fields)
        .with_default_fields(vec![TestFieldRegistry::title(), TestFieldRegistry::body()]);
    let mut scratch = PlannerScratch::new();

    let program = leit_query::term("rust");
    let plan = planner
        .plan_program(&program, &ctx, &mut scratch)
        .expect("plan multi-field term");

    match plan.program.get(plan.program.root()) {
        Some(QueryNode::TermExpansion {
            children, fields, ..
        }) => {
            assert_eq!(children.len(), 2);
            assert_eq!(fields.len(), 2);
        }
        other => panic!("expected term expansion, got {other:?}"),
    }
}

#[test]
fn shared_diamond_dag_stops_at_node_budget() {
    // 40 levels of a two-child diamond sharing one child: exponential without
    // memoized depth traversal. Lowering duplicates shared subtrees, so the
    // node budget must trip exactly at the limit rather than after any blowup,
    // and the whole thing must stay well inside a generous wall-clock bound.
    let mut builder = QueryBuilder::new();
    let mut current = builder.term("rust");
    for _ in 0..40 {
        current = builder.and(vec![current, current]);
    }
    let program = builder.build().expect("build diamond program");

    let max_nodes = 256;
    let planner = Planner::new().with_max_depth(64).with_max_nodes(max_nodes);
    let dictionary = TestDictionary;
    let fields = TestFieldRegistry;
    let mut scratch = PlannerScratch::new();
    let start = std::time::Instant::now();
    let result = planner.plan_program(&program, &context(&dictionary, &fields), &mut scratch);
    let elapsed = start.elapsed();
    assert_eq!(
        result,
        Err(QueryError::MaxNodesExceeded {
            max_nodes,
            actual_nodes: max_nodes + 1,
        }),
        "lowering must stop at the node budget"
    );
    assert!(
        elapsed < core::time::Duration::from_secs(5),
        "diamond DAG planning took {elapsed:?}"
    );
}

#[test]
fn long_chain_rejected_without_overflow() {
    // A 100k-deep NOT chain must be rejected with MaxDepthExceeded, not
    // overflow the stack during depth traversal.
    let mut builder = QueryBuilder::new();
    let mut current = builder.term("rust");
    for _ in 0..100_000 {
        current = builder.not(current);
    }
    let program = builder.build().expect("build chain program");

    let err = plan_it(&program).expect_err("depth guard must trip");
    assert!(matches!(err, QueryError::MaxDepthExceeded { .. }));
}

#[test]
fn nan_boost_is_rejected() {
    let mut builder = QueryBuilder::new();
    let term = builder.term("rust");
    builder.boost(term, f32::NAN);
    let program = builder.build().expect("build boosted program");
    let err = plan_it(&program).expect_err("NaN boost must be rejected");
    assert!(matches!(err, QueryError::InvalidBoost { .. }));
}

#[test]
fn infinite_boost_is_rejected() {
    let mut builder = QueryBuilder::new();
    let term = builder.term("rust");
    builder.boost(term, f32::INFINITY);
    let program = builder.build().expect("build boosted program");
    let err = plan_it(&program).expect_err("infinite boost must be rejected");
    assert!(matches!(err, QueryError::InvalidBoost { .. }));
}

#[test]
fn negative_boost_is_rejected() {
    let mut builder = QueryBuilder::new();
    let term = builder.term("rust");
    builder.boost(term, -2.0);
    let program = builder.build().expect("build boosted program");
    let err = plan_it(&program).expect_err("negative boost must be rejected");
    assert!(matches!(err, QueryError::InvalidBoost { .. }));
}

#[test]
fn composed_boost_overflow_is_rejected() {
    // Each factor is finite, but the composed product overflows to infinity.
    let mut builder = QueryBuilder::new();
    let term = builder.term("rust");
    let inner = builder.boost(term, 1.0e30);
    builder.boost(inner, 1.0e30);
    let program = builder.build().expect("build boosted program");
    let err = plan_it(&program).expect_err("composed infinite boost must be rejected");
    assert!(matches!(err, QueryError::InvalidBoost { .. }));
}
