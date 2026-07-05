// TODO: reemplazar anyhow con thiserror
use anyhow::Context;
use indicatif::ProgressIterator;
use llm::{
    LLMProvider,
    builder::{LLMBackend, LLMBuilder},
};
use serde::Serialize;

use crate::intake::dataset::DataSet;
pub struct EmbeddingProvider {
    pub model: Box<dyn LLMProvider>,
}

impl EmbeddingProvider {
    pub fn new(model: &str) -> anyhow::Result<Self> {
        let base_url = std::env::var("OLLAMA_URL").unwrap_or("http://127.0.0.1:11434".into());
        let llm = LLMBuilder::new()
            .backend(LLMBackend::Ollama)
            .base_url(base_url)
            .model(model)
            .build()
            .context("Error creando modelo embdding")?;
        Ok(EmbeddingProvider { model: llm })
    }
    pub fn new_openai(model: &str) -> anyhow::Result<Self> {
        dotenvy::dotenv().context(".env absent")?;
        let api_key = std::env::var("OPENAI_API_KEY").context("Api key absent")?;
        let llm = LLMBuilder::new()
            .backend(LLMBackend::OpenAI)
            .api_key(api_key)
            .model(model)
            .build()
            .context("Error creando modelo embdding")?;
        Ok(EmbeddingProvider { model: llm })
    }
    /// Embed every item in `dataset`, returning exactly one vector per item, in
    /// order.
    ///
    /// Items are sent to the backend in batches for throughput. Some backends
    /// (notably Ollama through `llm`) reject *multi-input* embedding requests
    /// even though each single input succeeds; when a batch request fails we
    /// transparently fall back to embedding that batch one item at a time. A
    /// genuine embedding failure is returned as an error rather than silently
    /// dropped — the previous behaviour skipped failed chunks, which left the
    /// returned vector shorter than (and misaligned with) the dataset and its
    /// payloads.
    pub async fn embed_properties<T>(&self, dataset: DataSet<T>) -> anyhow::Result<Vec<Vec<f32>>>
    where
        T: Serialize + Clone,
    {
        let data = dataset.data.context("There is no data to embed")?;
        let mut embeddings = Vec::with_capacity(data.len());
        for chunk in data.chunks(50).progress() {
            let chunk_strings: Vec<String> = chunk
                .iter()
                .map(|article| serde_json::to_string(article).context("Couldn't serialize article"))
                .collect::<anyhow::Result<_>>()?;

            match self.model.embed(chunk_strings.clone()).await {
                Ok(emb) => embeddings.extend(emb),
                Err(batch_err) => {
                    // Batch request failed (e.g. the Ollama backend can't send a
                    // multi-input embed): retry the chunk one item per request.
                    for (i, one) in chunk_strings.into_iter().enumerate() {
                        let single = self.model.embed(vec![one]).await.map_err(|e| {
                            anyhow::anyhow!(
                                "embedding failed on item {i} of a per-item fallback \
                                 (original batch error: {batch_err}): {e}"
                            )
                        })?;
                        embeddings.extend(single);
                    }
                }
            }
        }
        Ok(embeddings)
    }
}
