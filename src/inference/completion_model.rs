// TODO: reemplazar anyhow con thiserror
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
};

use anyhow::Context;
use indicatif::{ProgressBar, ProgressStyle};
use llm::{
    LLMProvider,
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
};
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::mpsc;

pub struct CompletionModel {
    pub model: Box<dyn LLMProvider>,
}

pub trait FieldEnhanceable {
    fn set_field(&mut self, modifications: String);
}

pub trait FieldEnhanceableG<T> {
    fn set_field(&mut self, modifications: T);
}
pub trait FieldsEnhanceable<T> {
    fn set_fields(&mut self, modifications: HashMap<String, Box<dyn FieldEnhanceableG<T>>>);
}

#[allow(dead_code)] // WIP: live-dumping wired into `perform_completion_and_live_dump`
struct LiveDumpFile {
    file: File,
}

#[allow(dead_code)] // WIP: live-dumping wired into `perform_completion_and_live_dump`
impl LiveDumpFile {
    fn create(filename: String) -> anyhow::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(filename.clone())
            .with_context(|| format!("Could not create file {filename}"))?;
        let mut file_dump = LiveDumpFile { file };
        file_dump
            .file
            .write_all("{\n".as_bytes())
            .context("Could not write while live dumping")?;
        Ok(file_dump)
    }
    fn append_property<T, M, S>(&mut self, property: T, _modifications: M) -> anyhow::Result<()>
    where
        T: Clone + Serialize,
    {
        let property_string =
            serde_json::to_string(&property).context("Could not serializa while live dumping")?;
        self.file
            .write_all(",\n".as_bytes())
            .context("Could not write while live dumping")?;
        self.file
            .write_all(property_string.as_bytes())
            .context("Could not write while live dumping")?;
        Ok(())
    }
    fn close<T>(mut self, _property: T) -> anyhow::Result<()>
    where
        T: Clone + Serialize,
    {
        self.file
            .write_all("}\n".as_bytes())
            .context("Could not write while live dumping")?;
        Ok(())
    }
}

//BUG: PRUEBAAA
impl CompletionModel {
    // TODO: Ver si esto se puede implementar de alguna otra manera.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> anyhow::Result<Self> {
        let base_url = std::env::var("OLLAMA_URL").unwrap_or("http://127.0.0.1:11434".into());
        let llm = LLMBuilder::new()
            .backend(LLMBackend::Ollama)
            .base_url(base_url)
            .model(model)
            .system(prompt)
            .build()
            .context("Error creando modelo embdding")?;

        Ok(CompletionModel { model: llm })
    }
    // TODO: Ver si esto se puede implementar de alguna otra manera.
    pub fn new_openai(model: impl Into<String>, prompt: impl Into<String>) -> anyhow::Result<Self> {
        dotenvy::dotenv().context(".env absent")?;
        let api_key = std::env::var("OPENAI_API_KEY").context("Api key absent")?;
        let llm = LLMBuilder::new()
            .backend(LLMBackend::OpenAI)
            .api_key(api_key)
            .model(model)
            .system(prompt)
            .build()
            .context("Error creando modelo embdding")?;
        Ok(CompletionModel { model: llm })
    }
    pub async fn perform_completion_and_live_dump<T>(
        &self,
        dataset: Vec<T>,
        _live_dump_file: Option<String>,
    ) -> anyhow::Result<Vec<T>>
    where
        T: Serialize + Clone,
    {
        // TODO: Eventualmente reemplazar con tracing / tracing_subscriber
        println!("Running completion");
        let mut generated_articles = vec![];
        let messages = |article: String| {
            vec![
                ChatMessage::assistant()
                    .content("Please provide me with the information I need for summarization")
                    .build(),
                ChatMessage::user()
                    .content("you will find it in the next message")
                    .build(),
                ChatMessage::user().content(article).build(),
            ]
        };
        let mut failed_ids = vec![];
        let pb = ProgressBar::new(dataset.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .expect("Failed to create progress style")
                .progress_chars("#>-"),
        );
        let mut tokens = 0u32;
        for (n, property) in dataset.into_iter().enumerate() {
            if n == 1 {
                println!("At least one iteration ran");
            }
            let property_string =
                serde_json::to_string(&property).context("Couldn't serialize property")?;
            let chat = self.model.chat(messages(property_string).as_slice()).await;
            match chat {
                Ok(response) => {
                    if let Some(text) = response.text() {
                        generated_articles.push(text.clone());
                    }
                    if let Some(usage) = response.usage() {
                        tokens += usage.total_tokens;
                        pb.set_message(format!("Total tokens so far: {}", tokens));
                    }
                }
                Err(_) => failed_ids.push(n),
            }
            pb.inc(1);
        }
        pb.finish();
        println!(
            "Returning from completion function. Failed chunk ids:\n{:#?}",
            failed_ids
        );
        todo!("finish this");
        // Ok(generated_articles)
    }
    pub async fn perform_completion<T>(&self, dataset: Vec<T>) -> anyhow::Result<Vec<String>>
    where
        T: Serialize,
    {
        // TODO: Eventualmente reemplazar con tracing / tracing_subscriber
        println!("Running completion");
        let mut generated_articles = vec![];
        let messages = |article: String| {
            vec![
                ChatMessage::assistant()
                    .content("Please provide me with the information I need for summarization")
                    .build(),
                ChatMessage::user()
                    .content("you will find it in the next message")
                    .build(),
                ChatMessage::user().content(article).build(),
            ]
        };
        let mut failed_ids = vec![];
        let pb = ProgressBar::new(dataset.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .expect("Failed to create progress style")
                .progress_chars("#>-"),
        );
        let mut tokens = 0u32;
        for (n, property) in dataset.into_iter().enumerate() {
            if n == 1 {
                println!("At least one iteration ran");
            }
            let property_string =
                serde_json::to_string(&property).context("Couldn't serialize property")?;
            let chat = self.model.chat(messages(property_string).as_slice()).await;
            match chat {
                Ok(response) => {
                    if let Some(text) = response.text() {
                        generated_articles.push(text.clone());
                    }
                    if let Some(usage) = response.usage() {
                        tokens += usage.total_tokens;
                        pb.set_message(format!("Total tokens so far: {}", tokens));
                    }
                }
                Err(_) => failed_ids.push(n),
            }
            pb.inc(1);
        }
        pb.finish();
        println!(
            "Returning from completion function. Failed chunk ids:\n{:#?}",
            failed_ids
        );
        Ok(generated_articles)
    }

    pub async fn perform_completion_dump_inelegant<T>(
        &self,
        dataset: Vec<T>,
        filename: String,
    ) -> anyhow::Result<Vec<String>>
    where
        T: Serialize + Clone + Sync + Send + 'static + DeserializeOwned + FieldEnhanceable,
    {
        // TODO: Eventualmente reemplazar con tracing / tracing_subscriber
        let (emiter, mut receiver) = mpsc::channel::<Vec<T>>(4);

        let handle = tokio::spawn(async move {
            while let Some(msg) = receiver.recv().await {
                dump_on_each_iteration(&msg, filename.clone())
                    .await
                    .context("Could not dump")
                    .unwrap();
            }
        });
        println!("Running completion");
        let mut generated_articles = vec![];
        let mut generated_objects = vec![];
        let messages = |article: String| {
            vec![
                ChatMessage::assistant()
                    .content("Please provide me with the information I need for summarization")
                    .build(),
                ChatMessage::user()
                    .content("you will find it in the next message")
                    .build(),
                ChatMessage::user().content(article).build(),
            ]
        };

        let mut failed_ids = vec![];
        let pb = ProgressBar::new(dataset.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .expect("Failed to create progress style")
                .progress_chars("#>-"),
        );
        let mut tokens = 0u32;
        for (n, mut property) in dataset.into_iter().enumerate() {
            if n == 1 {
                println!("At least one iteration ran");
            }
            let property_string =
                serde_json::to_string(&property).context("Couldn't serialize property")?;
            let chat = self.model.chat(messages(property_string).as_slice()).await;
            match chat {
                Ok(response) => {
                    if let Some(text) = response.text() {
                        property.set_field(text.clone());
                        generated_objects.push(property);
                        emiter
                            .send(generated_objects.clone())
                            .await
                            .context("Could not send")?;
                        generated_articles.push(text.clone());
                    }
                    if let Some(usage) = response.usage() {
                        tokens += usage.total_tokens;
                        pb.set_message(format!("Total tokens so far: {}", tokens));
                    }
                }
                Err(_) => failed_ids.push(n),
            }
            pb.inc(1);
        }
        drop(emiter);
        pb.finish();
        println!(
            "Returning from completion function. Failed chunk ids:\n{:#?}",
            failed_ids
        );
        handle.await.unwrap();
        Ok(generated_articles)
    }
}
pub async fn dump_on_each_iteration<T>(
    data: &Vec<T>,
    filename: impl Into<String>,
) -> anyhow::Result<()>
where
    T: Serialize + Clone,
{
    let data_string = serde_json::to_string(data).context("Could not serialize")?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(filename.into())
        .context("Could not create/overwrite dump file")?;
    file.write_all(data_string.as_bytes())
        .context("Could not dump")?;
    Ok(())
}
