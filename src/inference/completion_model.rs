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
use tokio::{sync::mpsc, task::JoinHandle};

/// A chat model that transforms records before they are embedded: summaries,
/// keywords, translations, or any other text you can ask an LLM for.
///
/// The prompt given to [`new`](Self::new) or [`new_openai`](Self::new_openai)
/// is the system prompt, so it holds your instructions. Each record is
/// serialized as JSON and sent in its own chat, one record at a time. Every
/// chat also carries two fixed messages that introduce the record as input for
/// summarization, whatever the system prompt asks for.
///
/// - [`perform_completion`](Self::perform_completion) returns the response
///   texts.
/// - [`perform_completion_dump_inelegant`](Self::perform_completion_dump_inelegant)
///   stores each response in its record through [`FieldEnhanceable`] and saves
///   the updated records to a JSON file as it goes.
///
/// Records that fail are left out of the results instead of returning an
/// error, so compare the number of responses with the number of records.
///
/// # Example
///
/// Summarize articles, then embed each title with its summary:
///
/// ```no_run
/// use lvv::inference::{CompletionModel, EmbeddingProvider};
/// use lvv::intake::dataset::DataSet;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct Article {
///     title: String,
///     body: String,
/// }
///
/// # async fn example(articles: Vec<Article>) -> anyhow::Result<()> {
/// let model = CompletionModel::new(
///     "llama3.2",
///     "Summarize the article in one sentence. Reply with the sentence only.",
/// )?;
/// let summaries = model
///     .perform_completion(articles.iter().collect::<Vec<_>>())
///     .await?;
/// // Failed records are left out, so check before pairing.
/// anyhow::ensure!(summaries.len() == articles.len(), "some completions failed");
///
/// let texts: Vec<String> = articles
///     .iter()
///     .zip(&summaries)
///     .map(|(article, summary)| format!("{}\n{summary}", article.title))
///     .collect();
/// let vectors = EmbeddingProvider::new("nomic-embed-text")?
///     .embed_properties(DataSet::new("articles", "summaries", texts))
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// With the `derive` feature, store the response in a field marked
/// `#[lvv(description)]` instead; see
/// [`perform_completion_dump_inelegant`](Self::perform_completion_dump_inelegant).
pub struct CompletionModel {
    /// Configured provider implementation.
    pub model: Box<dyn LLMProvider>,
}

/// Receives the LLM response for one record.
///
/// [`CompletionModel::perform_completion_dump_inelegant`] calls
/// [`set_field`](Self::set_field) with each record's response. Store it in the
/// field that should hold the result.
pub trait FieldEnhanceable {
    /// Stores the response text in the record.
    fn set_field(&mut self, modifications: String);
}

/// Applies a generated value of type `T` to one field of a value.
///
/// Not used by [`CompletionModel`] yet.
pub trait FieldEnhanceableG<T> {
    /// Updates the implementation-defined field.
    fn set_field(&mut self, modifications: T);
}
/// Applies generated values to several named fields.
///
/// Not used by [`CompletionModel`] yet.
pub trait FieldsEnhanceable<T> {
    /// Updates fields identified by the keys of `modifications`.
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
    /// Creates an Ollama chat client whose system prompt is `prompt`.
    ///
    /// The endpoint is read from `OLLAMA_URL` and defaults to
    /// `http://127.0.0.1:11434`. The model isn't contacted until a completion
    /// runs, so a wrong model name shows up then, as failed records.
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
    /// Creates an OpenAI chat client whose system prompt is `prompt`.
    ///
    /// Reads `OPENAI_API_KEY` after loading a `.env` file from the current
    /// directory or one of its parents.
    ///
    /// # Errors
    ///
    /// Returns an error if no `.env` file is found, even when `OPENAI_API_KEY`
    /// is already set, or if `OPENAI_API_KEY` is missing.
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
    /// Completes every serialized item while preparing incremental output.
    ///
    /// # Panics
    ///
    /// This method is unfinished and currently always panics after contacting
    /// the provider. Use [`Self::perform_completion`] instead.
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
    /// Sends each record to the model and returns the response texts, in record
    /// order.
    ///
    /// Records are sent one at a time, each serialized as JSON in its own chat.
    /// A progress bar shows the tokens used so far, and the indices of failed
    /// records are printed at the end.
    ///
    /// Records that fail, or whose response has no text, are left out of the
    /// result instead of returning an error. A wrong model name, for example,
    /// fails every record and returns `Ok` with an empty vector. Compare the
    /// lengths before pairing responses with records.
    ///
    /// See [`CompletionModel`] for an example.
    ///
    /// # Errors
    ///
    /// Returns an error only if a record can't be serialized as JSON.
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

    /// Sends each record to the model, stores each response in its record with
    /// [`FieldEnhanceable::set_field`], and saves the updated records to
    /// `filename`.
    ///
    /// After every successful record, `filename` is rewritten as a JSON array
    /// of all the records updated so far, so an interrupted run keeps its
    /// progress. The updated records are only available from that file: the
    /// method returns the response texts, like
    /// [`perform_completion`](Self::perform_completion). Failed records are
    /// left out of both.
    ///
    /// # Example
    ///
    /// A summary written by the model becomes part of the text that a derived
    /// [`VectorDatabaseItem`](crate::transform::transform::VectorDatabaseItem)
    /// embeds:
    ///
    #[cfg_attr(feature = "derive", doc = "```no_run")]
    #[cfg_attr(not(feature = "derive"), doc = "```ignore")]
    /// use lvv::inference::{CompletionModel, completion_model::FieldEnhanceable};
    /// use lvv::transform::transform::VectorDatabaseItem;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Clone, Serialize, Deserialize, VectorDatabaseItem)]
    /// struct Article {
    ///     #[lvv(description)]
    ///     title: String,
    ///     body: String,
    ///     // Filled in by the model, then embedded with the title.
    ///     #[lvv(description)]
    ///     summary: Option<String>,
    /// }
    ///
    /// impl FieldEnhanceable for Article {
    ///     fn set_field(&mut self, response: String) {
    ///         self.summary = Some(response);
    ///     }
    /// }
    ///
    /// # async fn example(articles: Vec<Article>) -> anyhow::Result<()> {
    /// let model = CompletionModel::new(
    ///     "llama3.2",
    ///     "Summarize the article in one sentence. Reply with the sentence only.",
    /// )?;
    /// model
    ///     .perform_completion_dump_inelegant(articles, "articles.json".into())
    ///     .await?;
    ///
    /// let summarized: Vec<Article> =
    ///     serde_json::from_str(&std::fs::read_to_string("articles.json")?)?;
    /// for article in &summarized {
    ///     // The description is "<title>\n<summary>".
    ///     let draft = article.try_into_database_item()?;
    ///     println!("{}", draft.description);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a record can't be serialized as JSON.
    ///
    /// # Panics
    ///
    /// If `filename` can't be written, this method panics or returns an error,
    /// depending on when the write fails.
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

        let handle: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
            while let Some(msg) = receiver.recv().await {
                dump_on_each_iteration(&msg, filename.clone())
                    .await
                    .context("Could not dump")?;
            }

            Ok(())
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
        handle.await??;
        Ok(generated_articles)
    }
}
/// Serializes `data` to `filename`, replacing any previous contents.
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
