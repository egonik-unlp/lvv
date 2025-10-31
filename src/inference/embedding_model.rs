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
    pub async fn embed_properties(&self, dataset: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut embeddings = vec![];

        let mut failed_ids = vec![];
        let mut log_emb = OpenOptions::new()
            .append(true)
            .create(true)
            .open("logs_taca_full.log")
            .context("Failed creating log file")?;
        for (n, chunk) in dataset.chunks(50).enumerate() {
            println!("Embedding chunk n = {}", n);
            let properties_string_chunk = chunk.to_vec();
            let embeddings_chunk = self.model.embed(properties_string_chunk.clone()).await;
            match embeddings_chunk {
                Ok(emb) => embeddings.extend(emb),
                Err(err) => {
                    failed_ids.push(n);
                    println!("Error con chunk = {}\n{}", n, err);
                    properties_string_chunk
                        .iter()
                        .map(|word| format!("{word},\n"))
                        .for_each(|word| log_emb.write_all(word.as_bytes()).unwrap());
                }
            }
        }
        println!(
            "Returning from embedding function. Failed chunk ids:\n{:#?}",
            failed_ids
        );
        Ok(embeddings)
    }
}
use crate::intake::dataset::DataSet;
use anyhow::Context;
use llm::{
    LLMProvider,
    builder::{LLMBackend, LLMBuilder},
};
use std::io::Write;
use std::{fs::OpenOptions, time::Duration};
use tokio::time::sleep;
