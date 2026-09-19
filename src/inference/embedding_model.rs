use std::sync::Arc;

use serde::Serialize;

use crate::{
    backend::{BackendError, BackendIdentity, EmbedBackend, OllamaClient, OpenAiCompatClient},
    intake::dataset::DataSet,
};

/// How many inputs go in one embedding request.
const BATCH_SIZE: usize = 50;

/// Why an embedding call failed.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The dataset's `data` is `None`.
    #[error("the dataset has no data to embed")]
    NoData,
    /// A record couldn't be serialized as JSON.
    #[error("could not serialize record {index} as JSON: {source}")]
    Serialize {
        /// The record's position in the input.
        index: usize,
        /// The serialization error.
        #[source]
        source: serde_json::Error,
    },
    /// The backend failed on an input, even when sent on its own.
    #[error("embedding input {index} failed: {source}")]
    Backend {
        /// The input's position.
        index: usize,
        /// The backend's error.
        #[source]
        source: BackendError,
    },
    /// The backend returned a different number of vectors than inputs.
    #[error("the backend returned {got} vectors for {expected} inputs")]
    CountMismatch {
        /// Inputs sent.
        expected: usize,
        /// Vectors received.
        got: usize,
    },
}

/// Embeds texts or records with an [`EmbedBackend`], one vector per input, in
/// input order.
///
/// Use [`embed_texts`](Self::embed_texts) for text you want embedded as is,
/// such as a [`VectorPointDraft`](crate::points::VectorPointDraft)'s
/// description, and [`embed_properties`](Self::embed_properties) to embed each
/// record's JSON.
///
/// # Example
///
/// ```no_run
/// use lvv::inference::EmbeddingProvider;
/// # async fn example() -> Result<(), lvv::inference::EmbedError> {
/// let provider = EmbeddingProvider::new("nomic-embed-text");
/// let vectors = provider.embed_texts(&["one", "two"]).await?;
/// assert_eq!(vectors.len(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct EmbeddingProvider {
    backend: Arc<dyn EmbedBackend>,
}

impl std::fmt::Debug for EmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingProvider")
            .field("backend", &self.backend.identity())
            .finish()
    }
}

impl EmbeddingProvider {
    /// An Ollama model, at `OLLAMA_URL` or `http://127.0.0.1:11434`.
    pub fn new(model: &str) -> Self {
        Self::from_backend(Arc::new(OllamaClient::new(model)))
    }

    /// An OpenAI model, with the key from `OPENAI_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns an error when no key is found; see
    /// [`OpenAiCompatClient::openai`].
    pub fn openai(model: &str) -> Result<Self, BackendError> {
        Ok(Self::from_backend(Arc::new(OpenAiCompatClient::openai(
            model,
        )?)))
    }

    /// A model on any OpenAI-compatible server.
    pub fn openai_compatible(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        Self::from_backend(Arc::new(OpenAiCompatClient::new(base_url, model, api_key)))
    }

    /// Any [`EmbedBackend`].
    pub fn from_backend(backend: Arc<dyn EmbedBackend>) -> Self {
        EmbeddingProvider { backend }
    }

    /// The provider, endpoint and model in use.
    pub fn identity(&self) -> BackendIdentity {
        self.backend.identity()
    }

    /// Embeds each text exactly as given, returning one vector per text, in
    /// order.
    ///
    /// Texts are sent in batches of 50. When a backend rejects a batch, its
    /// texts are sent one at a time before giving up, since some servers
    /// reject multi-input requests that they accept one by one.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first text that fails on its own.
    pub async fn embed_texts<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for (batch_number, chunk) in texts.chunks(BATCH_SIZE).enumerate() {
            let offset = batch_number * BATCH_SIZE;
            let inputs: Vec<&str> = chunk.iter().map(AsRef::as_ref).collect();
            let batch = match self.backend.embed(&inputs).await {
                Ok(vectors) if vectors.len() == inputs.len() => vectors,
                Ok(vectors) if inputs.len() == 1 => {
                    return Err(EmbedError::CountMismatch {
                        expected: 1,
                        got: vectors.len(),
                    });
                }
                Err(err) if inputs.len() == 1 => {
                    return Err(EmbedError::Backend {
                        index: offset,
                        source: err,
                    });
                }
                outcome => {
                    tracing::debug!(
                        batch = batch_number,
                        error = ?outcome.err(),
                        "batch embedding failed, retrying one input at a time"
                    );
                    self.embed_one_by_one(&inputs, offset).await?
                }
            };
            embeddings.extend(batch);
            tracing::debug!(
                done = embeddings.len(),
                total = texts.len(),
                "embedded batch"
            );
        }
        Ok(embeddings)
    }

    async fn embed_one_by_one(
        &self,
        inputs: &[&str],
        offset: usize,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut vectors = Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            let mut single =
                self.backend
                    .embed(&[input])
                    .await
                    .map_err(|source| EmbedError::Backend {
                        index: offset + i,
                        source,
                    })?;
            if single.len() != 1 {
                return Err(EmbedError::CountMismatch {
                    expected: 1,
                    got: single.len(),
                });
            }
            vectors.push(single.remove(0));
        }
        Ok(vectors)
    }

    /// Embeds each record's JSON serialization, returning one vector per
    /// record, in order.
    ///
    /// # Errors
    ///
    /// Returns an error when the dataset has no data, a record can't be
    /// serialized, or embedding fails as in
    /// [`embed_texts`](Self::embed_texts).
    pub async fn embed_properties<T>(
        &self,
        dataset: DataSet<T>,
    ) -> Result<Vec<Vec<f32>>, EmbedError>
    where
        T: Serialize,
    {
        let data = dataset.data.ok_or(EmbedError::NoData)?;
        let texts = data
            .iter()
            .enumerate()
            .map(|(index, record)| {
                serde_json::to_string(record)
                    .map_err(|source| EmbedError::Serialize { index, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.embed_texts(&texts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::{MockEmbed, http_error};

    fn provider(mock: &Arc<MockEmbed>) -> EmbeddingProvider {
        EmbeddingProvider::from_backend(mock.clone())
    }

    #[tokio::test]
    async fn texts_are_sent_as_is() {
        let mock = Arc::new(MockEmbed::lengths());
        provider(&mock)
            .embed_texts(&["Researcher\nUNLP"])
            .await
            .unwrap();
        assert_eq!(mock.batches(), vec![vec!["Researcher\nUNLP".to_string()]]);
    }

    #[tokio::test]
    async fn records_are_sent_as_json() {
        let mock = Arc::new(MockEmbed::lengths());
        let dataset = DataSet::new("memory", "t", vec![serde_json::json!({"title": "Rust"})]);
        provider(&mock).embed_properties(dataset).await.unwrap();
        assert_eq!(
            mock.batches(),
            vec![vec![r#"{"title":"Rust"}"#.to_string()]]
        );
    }

    #[tokio::test]
    async fn inputs_go_in_batches_of_fifty() {
        let mock = Arc::new(MockEmbed::lengths());
        let texts: Vec<String> = (0..120).map(|i| "x".repeat(i)).collect();
        let vectors = provider(&mock).embed_texts(&texts).await.unwrap();
        let sizes: Vec<usize> = mock.batches().iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![50, 50, 20]);
        assert_eq!(vectors.len(), 120);
        assert!(vectors.iter().enumerate().all(|(i, v)| v[0] == i as f32));
    }

    #[tokio::test]
    async fn rejected_batches_fall_back_to_single_inputs_in_order() {
        let mock = Arc::new(MockEmbed::new(|inputs| {
            if inputs.len() > 1 {
                Err(http_error(400))
            } else {
                Ok(vec![vec![inputs[0].len() as f32]])
            }
        }));
        let texts = ["a", "bb", "ccc", "dddd", "eeeee"];
        let vectors = provider(&mock).embed_texts(&texts).await.unwrap();
        assert_eq!(
            vectors,
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]
        );
    }

    #[tokio::test]
    async fn a_genuine_failure_names_the_input() {
        let mock = Arc::new(MockEmbed::new(|inputs| {
            if inputs.contains(&"bad") {
                Err(http_error(400))
            } else {
                Ok(inputs.iter().map(|_| vec![0.0]).collect())
            }
        }));
        let err = provider(&mock)
            .embed_texts(&["ok", "fine", "bad", "ok"])
            .await
            .unwrap_err();
        assert!(matches!(err, EmbedError::Backend { index: 2, .. }), "{err}");
    }

    #[tokio::test]
    async fn a_short_reply_is_retried_one_by_one() {
        let mock = Arc::new(MockEmbed::new(|inputs| {
            Ok(vec![vec![0.0]; inputs.len().min(1)])
        }));
        let vectors = provider(&mock).embed_texts(&["a", "b"]).await.unwrap();
        assert_eq!(vectors.len(), 2);
    }
}
