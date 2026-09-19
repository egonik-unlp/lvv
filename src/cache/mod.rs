//! Persistent caching for generated embeddings.

use sha2::{Digest, Sha256};

/// JSON-serializable embedding cache.
pub mod cache_embeddings;

/// A stable hex SHA-256 of `parts`. Each part is prefixed with its length, so
/// `["ab", "c"]` and `["a", "bc"]` hash differently.
pub(crate) fn hash_key<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::hash_key;

    #[test]
    fn parts_are_length_prefixed() {
        let ab_c = hash_key([b"ab".as_slice(), b"c".as_slice()]);
        let a_bc = hash_key([b"a".as_slice(), b"bc".as_slice()]);
        assert_ne!(ab_c, a_bc);
        assert_eq!(ab_c.len(), 64);
        assert_eq!(ab_c, hash_key([b"ab".as_slice(), b"c".as_slice()]));
    }
}
