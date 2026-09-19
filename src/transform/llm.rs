use std::sync::Arc;

use super::{
    definition::Transform,
    run::{Complete, Options, Run},
};
use crate::backend::{
    BackendError, BackendIdentity, ChatBackend, OllamaClient, OpenAiCompatClient,
};

/// A chat model to run [`Transform`]s with.
///
/// `Llm` only says where the model is; what to ask lives in the
/// [`Transform`]. Cloning is cheap, and one `Llm` can run any number of
/// transforms.
///
/// # Example
///
/// ```no_run
/// use lvv::transform::{Llm, Transform};
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct Article {
///     title: String,
///     body: String,
///     summary: Option<String>,
/// }
///
/// # async fn example(mut articles: Vec<Article>) -> Result<(), Box<dyn std::error::Error>> {
/// let llm = Llm::ollama("llama3.2");
/// let summarize = Transform::text("Summarize the article in one sentence.")
///     .input(|a: &Article| a.body.clone())
///     .apply(|a: &mut Article, summary| a.summary = Some(summary));
///
/// let report = llm
///     .run(&summarize, &mut articles)
///     .concurrency(4)
///     .cache("summaries.jsonl")
///     .await?;
/// for (index, error) in report.failures() {
///     eprintln!("article {index}: {error}");
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Llm {
    backend: Arc<dyn ChatBackend>,
}

impl std::fmt::Debug for Llm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Llm")
            .field("backend", &self.backend.identity())
            .finish()
    }
}

impl Llm {
    /// An Ollama model, at `OLLAMA_URL` or `http://127.0.0.1:11434`.
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::from_backend(Arc::new(OllamaClient::new(model)))
    }

    /// An OpenAI model, with the key from `OPENAI_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns an error when no key is found; see
    /// [`OpenAiCompatClient::openai`].
    pub fn openai(model: impl Into<String>) -> Result<Self, BackendError> {
        Ok(Self::from_backend(Arc::new(OpenAiCompatClient::openai(
            model,
        )?)))
    }

    /// A model on any OpenAI-compatible server, such as vLLM or LM Studio.
    pub fn openai_compatible(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self::from_backend(Arc::new(OpenAiCompatClient::new(base_url, model, api_key)))
    }

    /// Any [`ChatBackend`], such as a configured [`OllamaClient`].
    pub fn from_backend(backend: Arc<dyn ChatBackend>) -> Self {
        Llm { backend }
    }

    /// The provider, endpoint and model in use.
    pub fn identity(&self) -> BackendIdentity {
        self.backend.identity()
    }

    /// Runs `transform` over `records` and applies each output to its record.
    ///
    /// Every record gets an [`Outcome`](super::Outcome) in the returned
    /// [`Report`](super::Report), in record order: a record whose request
    /// fails is left unchanged and reported, and doesn't affect the others.
    /// Set options on the returned [`Run`], then `.await` it.
    ///
    /// # Errors
    ///
    /// The run as a whole fails only when it can't go on: the transform has
    /// no [`apply`](Transform::apply) step, the cache can't be read or
    /// written, or the circuit breaker opens. Records completed before that
    /// keep their outputs, and the error says how many there were.
    pub fn run<'a, T, O>(
        &'a self,
        transform: &'a Transform<T, O>,
        records: &'a mut [T],
    ) -> Run<'a, T, O> {
        Run {
            backend: self.backend.as_ref(),
            transform,
            records,
            options: Options::default(),
        }
    }

    /// Runs `transform` over `records` and returns the outputs instead of
    /// applying them: one result per record, in record order.
    ///
    /// Takes the same options as [`run`](Self::run), and fails as a whole in
    /// the same cases, except that no [`apply`](Transform::apply) step is
    /// needed.
    pub fn complete<'a, T, O>(
        &'a self,
        transform: &'a Transform<T, O>,
        records: &'a [T],
    ) -> Complete<'a, T, O> {
        Complete {
            backend: self.backend.as_ref(),
            transform,
            records,
            options: Options::default(),
        }
    }
}
