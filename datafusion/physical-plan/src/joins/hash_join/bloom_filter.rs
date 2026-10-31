// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! A Bloom filter over the join key hashes of every build partition of a hash join.

use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::{DataType, Schema};
use datafusion_common::Result;
use datafusion_common::hash_utils::with_hashes;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr_common::physical_expr::{
    DynHash, PhysicalExpr, PhysicalExprRef,
};

use crate::joins::hash_join::partitioned_hash_eval::SeededRandomState;

/// The bits per key of a [`JoinKeyBloomFilter`].
const BITS_PER_KEY: usize = 12;

/// A register-blocked Bloom filter over join key hashes: each key sets three bits of one 64-bit
/// word, so probing a key reads one word.
///
/// Probing a Bloom filter costs one memory read per row, where looking a key up in the hash table
/// of a join follows the chain of its bucket. A partitioned join can publish one filter for all its
/// build partitions, where a lookup must first route each row to the hash table of its partition.
#[derive(Debug)]
pub struct JoinKeyBloomFilter {
    words: Vec<u64>,
    /// The number of words minus one; the number of words is a power of two.
    mask: u64,
}

impl JoinKeyBloomFilter {
    /// A filter sized for `keys` keys.
    pub fn with_capacity(keys: usize) -> Self {
        let words = (keys.max(1) * BITS_PER_KEY / 64).next_power_of_two().max(1);
        Self {
            words: vec![0; words],
            mask: words as u64 - 1,
        }
    }

    /// The word of `hash` and the three bits it sets in that word.
    #[inline]
    fn word_and_bits(&self, hash: u64) -> (usize, u64) {
        // The high bits pick the word, as the low bits pick the hash table bucket of the join.
        let word = ((hash >> 32) & self.mask) as usize;
        let bits = (1u64 << (hash & 63))
            | (1u64 << ((hash >> 6) & 63))
            | (1u64 << ((hash >> 12) & 63));
        (word, bits)
    }

    /// Adds `hash` to the filter.
    pub fn insert(&mut self, hash: u64) {
        let (word, bits) = self.word_and_bits(hash);
        self.words[word] |= bits;
    }

    /// Whether `hash` may have been added to the filter.
    #[inline]
    pub fn contains(&self, hash: u64) -> bool {
        let (word, bits) = self.word_and_bits(hash);
        self.words[word] & bits == bits
    }
}

/// Physical expression that checks whether the join keys of a row may be in a
/// [`JoinKeyBloomFilter`]. A row whose keys are in the build side always passes; a row whose keys
/// are not passes with the false positive rate of the filter.
pub struct BloomFilterExpr {
    /// Columns in the ON clause used to compute the join key
    on_columns: Vec<PhysicalExprRef>,
    /// The random state the build side hashed its keys with
    random_state: SeededRandomState,
    filter: Arc<JoinKeyBloomFilter>,
}

impl BloomFilterExpr {
    /// Create a new BloomFilterExpr
    pub fn new(
        on_columns: Vec<PhysicalExprRef>,
        random_state: SeededRandomState,
        filter: Arc<JoinKeyBloomFilter>,
    ) -> Self {
        Self {
            on_columns,
            random_state,
            filter,
        }
    }
}

impl std::fmt::Debug for BloomFilterExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl Display for BloomFilterExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cols = self
            .on_columns
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "bloom_filter({cols})")
    }
}

impl Hash for BloomFilterExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.on_columns.dyn_hash(state);
        self.random_state.seed().hash(state);
        // The filter is compared by pointer, as HashJoinExec creates one per execution.
        Arc::as_ptr(&self.filter).hash(state);
    }
}

impl PartialEq for BloomFilterExpr {
    fn eq(&self, other: &Self) -> bool {
        self.on_columns == other.on_columns
            && self.random_state.seed() == other.random_state.seed()
            && Arc::ptr_eq(&self.filter, &other.filter)
    }
}

impl Eq for BloomFilterExpr {}

impl PhysicalExpr for BloomFilterExpr {
    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.on_columns.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(BloomFilterExpr::new(
            children,
            self.random_state.clone(),
            Arc::clone(&self.filter),
        )))
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let join_keys = self
            .on_columns
            .iter()
            .map(|c| c.evaluate(batch)?.into_array(batch.num_rows()))
            .collect::<Result<Vec<_>>>()?;
        with_hashes(&join_keys, self.random_state.random_state(), |hashes| {
            let kept = BooleanBuffer::collect_bool(hashes.len(), |row| {
                self.filter.contains(hashes[row])
            });
            Ok(ColumnarValue::Array(Arc::new(BooleanArray::new(
                kept, None,
            ))))
        })
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_every_inserted_hash() {
        let mut filter = JoinKeyBloomFilter::with_capacity(1000);
        for hash in (0..1000u64).map(|v| v.wrapping_mul(0x9E37_79B9_7F4A_7C15)) {
            filter.insert(hash);
        }
        assert!(
            (0..1000u64)
                .map(|v| v.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .all(|hash| filter.contains(hash))
        );
        let false_positives = (1000..101_000u64)
            .map(|v| v.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .filter(|hash| filter.contains(*hash))
            .count();
        assert!(false_positives < 6_000, "{false_positives} false positives");
    }
}
