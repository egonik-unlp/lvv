use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    BackendError, BackendIdentity, ChatBackend, ChatRequest, ChatResponse, DEFAULT_TIMEOUT,
    EmbedBackend, Usage, join_url, post_json,
};

/// The endpoint used when `OLLAMA_URL` isn't set.
pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";

/// A client for Ollama's native API, for both chat and embedding models.
///
/// # Example
///
/// ```
/// use lvv::backend::{ChatBackend, OllamaClient};
///
/// let client = OllamaClient::with_base_url("http://gpu-box:11434", "llama3.2");
/// assert_eq!(client.identity().model, "llama3.2");
/// ```
#[derive(Debug, Clone)]
pub struct OllamaClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
    timeout: Duration,
}

impl OllamaClient {
    /// A client for `model` at `OLLAMA_URL`, or at `http://127.0.0.1:11434`
    /// when that variable isn't set.
    ///
    /// Nothing is contacted until the first request, so a wrong model name
    /// shows up then, as a fatal [`BackendError`].
    pub fn new(model: impl Into<String>) -> Self {
        let base_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.into());
        Self::with_base_url(base_url, model)
    }

    /// A client for `model` at `base_url`, such as `http://gpu-box:11434`.
    pub fn with_base_url(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        OllamaClient {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// How long one request may take. Defaults to two minutes.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn chat_body(&self, request: &ChatRequest) -> Value {
        let mut body = json!({
            "model": self.model,
            "stream": false,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        });
        if let Some(schema) = &request.schema {
            body["format"] = schema.schema.clone();
        }
        body
    }
}

#[derive(Deserialize)]
struct ChatReply {
    message: Option<ReplyMessage>,
    prompt_eval_count: Option<u64>,
    eval_count: Option<u64>,
}

#[derive(Deserialize)]
struct ReplyMessage {
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct EmbedReply {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl ChatBackend for OllamaClient {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            kind: "ollama".into(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
        }
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BackendError> {
        let url = join_url(&self.base_url, "/api/chat");
        let reply: ChatReply = post_json(
            &self.client,
            &url,
            None,
            self.timeout,
            &self.chat_body(request),
        )
        .await?;
        let usage = match (reply.prompt_eval_count, reply.eval_count) {
            (None, None) => None,
            (prompt, completion) => Some(Usage {
                prompt_tokens: prompt.unwrap_or(0),
                completion_tokens: completion.unwrap_or(0),
            }),
        };
        Ok(ChatResponse {
            text: reply.message.map(|m| m.content).unwrap_or_default(),
            usage,
        })
    }
}

#[async_trait]
impl EmbedBackend for OllamaClient {
    fn identity(&self) -> BackendIdentity {
        ChatBackend::identity(self)
    }

    async fn embed(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, BackendError> {
        let url = join_url(&self.base_url, "/api/embed");
        let body = json!({"model": self.model, "input": inputs});
        let reply: EmbedReply = post_json(&self.client, &url, None, self.timeout, &body).await?;
        Ok(reply.embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ErrorKind, OutputSchema, stub};

    fn request(schema: Option<OutputSchema>) -> ChatRequest {
        ChatRequest {
            system: "Extract keywords".into(),
            user: "{\"title\":\"Rust\"}".into(),
            schema,
        }
    }

    #[test]
    fn chat_body_has_exactly_the_system_and_user_messages() {
        let client = OllamaClient::with_base_url("http://x", "llama3.2");
        let body = client.chat_body(&request(None));
        assert_eq!(
            body["messages"],
            json!([
                {"role": "system", "content": "Extract keywords"},
                {"role": "user", "content": "{\"title\":\"Rust\"}"},
            ])
        );
        assert_eq!(body["stream"], json!(false));
        assert!(body.get("format").is_none());
    }

    #[test]
    fn chat_body_sends_the_schema_as_format() {
        let client = OllamaClient::with_base_url("http://x", "llama3.2");
        let schema = json!({"type": "object"});
        let body = client.chat_body(&request(Some(OutputSchema {
            name: "Tags".into(),
            schema: schema.clone(),
        })));
        assert_eq!(body["format"], schema);
    }

    #[tokio::test]
    async fn chat_parses_text_and_usage() {
        let server = stub::StubServer::start(vec![stub::StubResponse::ok(
            r#"{"message":{"role":"assistant","content":"rust, vectors"},"prompt_eval_count":12,"eval_count":3}"#,
        )])
        .await;
        let client = OllamaClient::with_base_url(server.url(""), "llama3.2");
        let response = client.chat(&request(None)).await.unwrap();
        assert_eq!(response.text, "rust, vectors");
        assert_eq!(response.usage.unwrap().total(), 15);
        let sent = server.requests();
        assert_eq!(sent[0].path, "/api/chat");
        assert_eq!(sent[0].body["model"], json!("llama3.2"));
    }

    #[tokio::test]
    async fn unknown_model_is_fatal() {
        let server = stub::StubServer::start(vec![
            stub::StubResponse::status(404)
                .body(r#"{"error":"model \"nope\" not found, try pulling it first"}"#),
        ])
        .await;
        let client = OllamaClient::with_base_url(server.url(""), "nope");
        let err = client.chat(&request(None)).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Fatal);
        assert!(err.message.contains("not found"));
    }

    #[tokio::test]
    async fn embed_sends_raw_inputs() {
        let server = stub::StubServer::start(vec![stub::StubResponse::ok(
            r#"{"embeddings":[[0.5,1.0]]}"#,
        )])
        .await;
        let client = OllamaClient::with_base_url(server.url(""), "embeddinggemma");
        let vectors = client.embed(&["Researcher\nUNLP"]).await.unwrap();
        assert_eq!(vectors, vec![vec![0.5, 1.0]]);
        let sent = server.requests();
        assert_eq!(sent[0].path, "/api/embed");
        assert_eq!(sent[0].body["input"], json!(["Researcher\nUNLP"]));
    }
}
