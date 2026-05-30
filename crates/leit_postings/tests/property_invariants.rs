// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Property-based invariant tests for `leit_postings`.

use std::collections::BTreeMap;

use leit_core::TermId;
use leit_postings::{BlockCursorState, InMemoryPostings, Posting, PostingsList};
use proptest::collection::vec;
use proptest::prelude::*;

proptest! {
    #[test]
    fn postings_list_keeps_doc_ids_sorted_and_latest_value(
        entries in vec((any::<u32>(), 1_u32..100_u32), 1..40)
    ) {
        let term_id = TermId::new(0);
        let mut list = PostingsList::new(term_id);
        let mut expected = BTreeMap::new();

        for (doc_id, term_freq) in &entries {
            list.add(Posting {
                doc_id: *doc_id,
                term_freq: *term_freq,
                positions: None,
            });
            expected.insert(*doc_id, *term_freq);
        }

        let actual_doc_ids: Vec<_> = list.postings.iter().map(|posting| posting.doc_id).collect();
        let expected_doc_ids: Vec<_> = expected.keys().copied().collect();
        prop_assert_eq!(actual_doc_ids, expected_doc_ids);

        for posting in &list.postings {
            prop_assert_eq!(expected.get(&posting.doc_id).copied(), Some(posting.term_freq));
        }
    }

    #[test]
    fn seek_lands_on_the_first_doc_at_or_after_target(
        entries in vec((any::<u32>(), 1_u32..100_u32), 1..40),
        target in any::<u32>(),
    ) {
        let term_id = TermId::new(0);
        let mut list = PostingsList::new(term_id);
        let mut expected = BTreeMap::new();

        for (doc_id, term_freq) in &entries {
            list.add(Posting {
                doc_id: *doc_id,
                term_freq: *term_freq,
                positions: None,
            });
            expected.insert(*doc_id, *term_freq);
        }

        let expected_docs: Vec<_> = expected.keys().copied().collect();
        let mut postings = InMemoryPostings::new();
        postings.add(list);
        let mut cursor = postings.cursor(term_id).expect("cursor should exist");

        let found = cursor.seek(target);
        let expected_doc = expected_docs.iter().copied().find(|doc_id| *doc_id >= target);

        prop_assert_eq!(found, expected_doc.is_some());
        prop_assert_eq!(cursor.doc(), expected_doc);
        if let Some(doc_id) = expected_doc {
            prop_assert_eq!(cursor.term_freq(), expected[&doc_id]);
        }
    }
}

#[test]
fn in_memory_cursor_exposes_singleton_block_seam() {
    let term_id = TermId::new(7);
    let mut list = PostingsList::new(term_id);
    list.add(Posting {
        doc_id: 3_u32,
        term_freq: 2,
        positions: None,
    });
    list.add(Posting {
        doc_id: 9_u32,
        term_freq: 5,
        positions: None,
    });

    let mut postings = InMemoryPostings::new();
    postings.add(list);
    let mut cursor = postings.cursor(term_id).expect("cursor should exist");

    assert_eq!(
        cursor.block_state(),
        BlockCursorState::Ready {
            end_doc: 3_u32,
            max_term_freq: 2,
        }
    );
    assert!(cursor.advance_block());
    assert_eq!(cursor.doc(), Some(9_u32));
    assert_eq!(
        cursor.block_state(),
        BlockCursorState::Ready {
            end_doc: 9_u32,
            max_term_freq: 5,
        }
    );
    assert!(!cursor.advance_block());
    assert_eq!(cursor.block_state(), BlockCursorState::Exhausted);
}
