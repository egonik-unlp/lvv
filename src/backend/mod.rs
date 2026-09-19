//! HTTP clients for chat and embedding models.
//!
//! [`ChatBackend`] and [`EmbedBackend`] are the only things the rest of lvv
//! talks to. Two implementations ship with the crate:
//!
//! - [`OllamaClient`] speaks Ollama's native API (`/api/chat`, `/api/embed`).
//! - [`OpenAiCompatClient`] speaks the OpenAI API (`/chat/completions`,
//!   `/embeddings`), which OpenAI, vLLM, LM Studio, llama.cpp's server,
//!   OpenRouter, Groq and Ollama's `/v1` endpoint all accept.
//!
//! Implement the traits yourself to plug in another provider.
//!
//! Every failure is a [`BackendError`] whose [`ErrorKind`] says whether trying
//! again can help: rate limits, server errors, timeouts and connection
//! failures are [`ErrorKind::Retryable`]; authentication failures, unknown
//! models and rejected requests are [`ErrorKind::Fatal`].

use std::time::Duration;

use async_trait::async_trait;
use serde::{Serialize, de::DeserializeOwned};

/// Ollama's native API.
pub mod ollama;
/// The OpenAI API and compatible servers.
pub mod openai;

#[cfg(test)]
pub(crate) mod mock;
#[cfg(test)]
pub(crate) mod stub;

pub use ollama::OllamaClient;
pub use openai::OpenAiCompatClient;

/// Default time a single request may take before it fails as retryable.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Which provider, endpoint and model a backend talks to.
///
/// It is part of every completion cache key, so outputs from one model are
/// never reused for another.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendIdentity {
    /// The kind of API, such as `"ollama"` or `"openai"`.
    pub kind: String,
    /// The base URL requests go to.
    pub base_url: String,
    /// The model name.
    pub model: String,
}

/// A JSON schema the model's reply must follow.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSchema {
    /// A name for the schema, made of ASCII letters, digits, `_` and `-`.
    pub name: String,
    /// The JSON schema itself.
    pub schema: serde_json::Value,
}

/// One chat: a system prompt and one user message.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// The instructions.
    pub system: String,
    /// The input for this record.
    pub user: String,
    /// When set, the reply must be JSON that follows this schema.
    pub schema: Option<OutputSchema>,
}

/// Tokens a request used, when the backend reports them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u64,
    /// Tokens in the reply.
    pub completion_tokens: u64,
}

impl Usage {
    /// Prompt and reply tokens together.
    pub fn total(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// The model's reply to a [`ChatRequest`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    /// The reply text. Empty when the model returned no content.
    pub text: String,
    /// Token usage, when the backend reports it.
    pub usage: Option<Usage>,
}

/// Whether a failed request is worth trying again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Rate limits, server errors, timeouts and connection failures.
    Retryable,
    /// Authentication failures, unknown models, rejected requests and
    /// configuration errors. Trying again gives the same result.
    Fatal,
}

/// A failed request to a model backend.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{}{message}", status.map(|s| format!("HTTP {s}: ")).unwrap_or_default())]
pub struct BackendError {
    /// Whether trying again can help.
    pub kind: ErrorKind,
    /// The HTTP status, when the server answered.
    pub status: Option<u16>,
    /// What went wrong, including the server's message when there is one.
    pub message: String,
    /// How long the server asked to wait before trying again.
    pub retry_after: Option<Duration>,
}

impl BackendError {
    /// A fatal error that is not tied to an HTTP response, such as a missing
    /// API key.
    pub fn config(message: impl Into<String>) -> Self {
        BackendError {
            kind: ErrorKind::Fatal,
            status: None,
            message: message.into(),
            retry_after: None,
        }
    }

    /// Whether this error is [`ErrorKind::Retryable`].
    pub fn is_retryable(&self) -> bool {
        self.kind == ErrorKind::Retryable
    }

    fn from_status(status: u16, body: &str, retry_after: Option<Duration>) -> Self {
        let body = body.trim();
        let message = if body.is_empty() {
            "the server returned no details".to_string()
        } else {
            // Keep error messages readable when a server returns an HTML page.
            body.chars().take(500).collect()
        };
        BackendError {
            kind: classify_status(status),
            status: Some(status),
            message,
            retry_after,
        }
    }

    fn from_reqwest(err: reqwest::Error) -> Self {
        let kind = if err.is_builder() {
            ErrorKind::Fatal
        } else {
            // Timeouts, refused connections, resets and truncated bodies.
            ErrorKind::Retryable
        };
        BackendError {
            kind,
            status: err.status().map(|s| s.as_u16()),
            message: err.to_string(),
            retry_after: None,
        }
    }

    fn invalid_response(message: impl Into<String>) -> Self {
        // A 200 whose body isn't what the API promises: another server is
        // listening there, and trying again won't change that.
        BackendError {
            kind: ErrorKind::Fatal,
            status: None,
            message: message.into(),
            retry_after: None,
        }
    }
}

/// Classifies an HTTP error status: 408, 429 and 5xx are retryable, every
/// other status is fatal.
pub fn classify_status(status: u16) -> ErrorKind {
    match status {
        408 | 429 | 500..=599 => ErrorKind::Retryable,
        _ => ErrorKind::Fatal,
    }
}

/// A chat model that answers one [`ChatRequest`] at a time.
///
/// Implementations must be safe to call concurrently: a transform run with
/// concurrency above one calls [`chat`](Self::chat) from several tasks.
#[async_trait]
pub trait ChatBackend: Send + Sync {
    /// The provider, endpoint and model this backend talks to.
    fn identity(&self) -> BackendIdentity;
    /// Sends one chat and returns the reply.
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BackendError>;
}

/// An embedding model.
#[async_trait]
pub trait EmbedBackend: Send + Sync {
    /// The provider, endpoint and model this backend talks to.
    fn identity(&self) -> BackendIdentity;
    /// Embeds each input, returning one vector per input in the same order.
    async fn embed(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, BackendError>;
}

/// Sends a JSON `POST` and decodes a JSON reply, turning every failure into a
/// classified [`BackendError`].
pub(crate) async fn post_json<B, R>(
    client: &reqwest::Client,
    url: &str,
    bearer: Option<&str>,
    timeout: Duration,
    body: &B,
) -> Result<R, BackendError>
where
    B: Serialize + ?Sized,
    R: DeserializeOwned,
{
    let mut request = client.post(url).timeout(timeout).json(body);
    if let Some(token) = bearer {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.map_err(BackendError::from_reqwest)?;
    let status = response.status();
    if !status.is_success() {
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<u64>().ok())
            .map(Duration::from_secs);
        let body = response.text().await.unwrap_or_default();
        return Err(BackendError::from_status(
            status.as_u16(),
            &body,
            retry_after,
        ));
    }
    let bytes = response.bytes().await.map_err(BackendError::from_reqwest)?;
    serde_json::from_slice(&bytes).map_err(|err| {
        BackendError::invalid_response(format!(
            "unexpected response from {url}: {err}; body: {}",
            String::from_utf8_lossy(&bytes)
                .chars()
                .take(300)
                .collect::<String>()
        ))
    })
}

/// Joins a base URL and a path without doubling or dropping the slash.
pub(crate) fn join_url(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_are_classified() {
        for status in [408, 429, 500, 502, 503, 504] {
            assert_eq!(classify_status(status), ErrorKind::Retryable, "{status}");
        }
        for status in [400, 401, 403, 404, 422] {
            assert_eq!(classify_status(status), ErrorKind::Fatal, "{status}");
        }
    }

    #[test]
    fn urls_join_with_one_slash() {
        assert_eq!(join_url("http://x:1/", "/api/chat"), "http://x:1/api/chat");
        assert_eq!(
            join_url("http://x:1/v1", "chat/completions"),
            "http://x:1/v1/chat/completions"
        );
    }

    #[tokio::test]
    async fn http_errors_are_classified_end_to_end() {
        let client = reqwest::Client::new();
        let cases = [
            (
                429,
                Some("7"),
                ErrorKind::Retryable,
                Some(Duration::from_secs(7)),
            ),
            (503, None, ErrorKind::Retryable, None),
            (401, None, ErrorKind::Fatal, None),
            (404, None, ErrorKind::Fatal, None),
        ];
        for (status, retry_after, kind, expected_wait) in cases {
            let server = stub::StubServer::start(vec![
                stub::StubResponse::status(status)
                    .retry_after(retry_after)
                    .body("{\"error\":\"nope\"}"),
            ])
            .await;
            let err = post_json::<_, serde_json::Value>(
                &client,
                &server.url("/x"),
                None,
                DEFAULT_TIMEOUT,
                &serde_json::json!({}),
            )
            .await
            .unwrap_err();
            assert_eq!(err.kind, kind, "status {status}");
            assert_eq!(err.status, Some(status));
            assert_eq!(err.retry_after, expected_wait);
            assert!(err.message.contains("nope"));
        }
    }

    #[tokio::test]
    async fn timeouts_and_refused_connections_are_retryable() {
        let client = reqwest::Client::new();
        let silent = stub::StubServer::silent().await;
        let err = post_json::<_, serde_json::Value>(
            &client,
            &silent.url("/x"),
            None,
            Duration::from_millis(100),
            &serde_json::json!({}),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Retryable, "{err}");

        let closed = stub::closed_port_url().await;
        let err = post_json::<_, serde_json::Value>(
            &client,
            &closed,
            None,
            DEFAULT_TIMEOUT,
            &serde_json::json!({}),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Retryable, "{err}");
    }

    #[tokio::test]
    async fn an_unexpected_body_is_fatal() {
        let server = stub::StubServer::start(vec![stub::StubResponse::ok("<html>hi</html>")]).await;
        let err = post_json::<_, serde_json::Value>(
            &reqwest::Client::new(),
            &server.url("/x"),
            None,
            DEFAULT_TIMEOUT,
            &serde_json::json!({}),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Fatal);
    }
}
