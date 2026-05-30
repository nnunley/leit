// Copyright 2026 the Leit Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    deprecated,
    reason = "codec.rs is a deprecated legacy directory-format artifact; retained only for compatibility"
)]
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "encode_segment is the frozen legacy directory encoder; unused in the lib build (to_segment_bytes emits the new format) but exercised by the in-crate deprecated-shim test"
    )
)]

use alloc::vec::Vec;

use crate::error::IndexError;
use crate::memory::InMemoryIndex;
#[expect(
    deprecated,
    reason = "SectionKind is deprecated but still used in the legacy directory-format encoder"
)]
use crate::segment::{SectionKind, magic, version};

pub(crate) fn encode_segment(index: &InMemoryIndex) -> Result<Vec<u8>, IndexError> {
    let term_dictionary = encode_term_dictionary(index)?;
    let field_metadata = encode_field_metadata(index)?;
    let (postings_metadata, postings_payload) = encode_postings(index)?;

    let sections = [
        (SectionKind::TermDictionary, term_dictionary),
        (SectionKind::FieldMetadata, field_metadata),
        (SectionKind::PostingsMetadata, postings_metadata),
        (SectionKind::PostingsPayload, postings_payload),
    ];

    let section_count = u32::try_from(sections.len()).map_err(|_| IndexError::ValueOutOfRange)?;
    let directory_len = sections
        .len()
        .checked_mul(12)
        .ok_or(IndexError::ValueOutOfRange)?;
    let mut offset = 24_usize
        .checked_add(directory_len)
        .ok_or(IndexError::ValueOutOfRange)?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&magic());
    push_u16(&mut bytes, version());
    push_u16(&mut bytes, 0);
    push_u32(&mut bytes, index.document_count());
    push_u32(
        &mut bytes,
        u32::try_from(index.term_entries().len()).map_err(|_| IndexError::ValueOutOfRange)?,
    );
    push_u32(
        &mut bytes,
        u32::try_from(index.field_stats().len()).map_err(|_| IndexError::ValueOutOfRange)?,
    );
    push_u32(&mut bytes, section_count);

    for (kind, payload) in &sections {
        let len_u32 = u32::try_from(payload.len()).map_err(|_| IndexError::ValueOutOfRange)?;
        let offset_u32 = u32::try_from(offset).map_err(|_| IndexError::ValueOutOfRange)?;
        push_u32(&mut bytes, kind.as_u32());
        push_u32(&mut bytes, offset_u32);
        push_u32(&mut bytes, len_u32);
        offset = offset
            .checked_add(payload.len())
            .ok_or(IndexError::ValueOutOfRange)?;
    }

    for (_, payload) in sections {
        bytes.extend_from_slice(&payload);
    }

    Ok(bytes)
}

fn encode_term_dictionary(index: &InMemoryIndex) -> Result<Vec<u8>, IndexError> {
    let mut bytes = Vec::new();
    push_u32(
        &mut bytes,
        u32::try_from(index.term_entries().len()).map_err(|_| IndexError::ValueOutOfRange)?,
    );
    for entry in index.term_entries() {
        push_u32(&mut bytes, entry.field_id.as_u32());
        push_u32(&mut bytes, entry.term_id.as_u32());
        let raw = entry.term.as_bytes();
        push_u32(
            &mut bytes,
            u32::try_from(raw.len()).map_err(|_| IndexError::ValueOutOfRange)?,
        );
        bytes.extend_from_slice(raw);
    }
    Ok(bytes)
}

fn encode_field_metadata(index: &InMemoryIndex) -> Result<Vec<u8>, IndexError> {
    let mut bytes = Vec::new();
    push_u32(
        &mut bytes,
        u32::try_from(index.field_stats().len()).map_err(|_| IndexError::ValueOutOfRange)?,
    );
    for stats in index.field_stats().values() {
        push_u32(&mut bytes, stats.field_id.as_u32());
        push_u32(&mut bytes, stats.doc_count);
        push_u32(&mut bytes, stats.total_terms);
    }
    Ok(bytes)
}

fn encode_postings(index: &InMemoryIndex) -> Result<(Vec<u8>, Vec<u8>), IndexError> {
    let mut metadata = Vec::new();
    let mut payload = Vec::new();

    push_u32(
        &mut metadata,
        u32::try_from(index.postings().len()).map_err(|_| IndexError::ValueOutOfRange)?,
    );

    for (term_id, postings) in index.postings() {
        let offset = u32::try_from(payload.len()).map_err(|_| IndexError::ValueOutOfRange)?;
        push_u32(
            &mut payload,
            u32::try_from(postings.len()).map_err(|_| IndexError::ValueOutOfRange)?,
        );
        for posting in postings {
            push_u32(&mut payload, posting.doc_id);
            push_u32(&mut payload, posting.term_freq);
        }
        let len = u32::try_from(payload.len())
            .map_err(|_| IndexError::ValueOutOfRange)?
            .checked_sub(offset)
            .ok_or(IndexError::ValueOutOfRange)?;

        push_u32(&mut metadata, term_id.as_u32());
        push_u32(&mut metadata, offset);
        push_u32(&mut metadata, len);
        push_u32(
            &mut metadata,
            u32::try_from(postings.len()).map_err(|_| IndexError::ValueOutOfRange)?,
        );
    }

    Ok((metadata, payload))
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
