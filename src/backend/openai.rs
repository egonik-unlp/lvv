use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    BackendError, BackendIdentity, ChatBackend, ChatRequest, ChatResponse, DEFAULT_TIMEOUT,
    EmbedBackend, Usage, join_url, post_json,
};

/// OpenAI's API endpoint.
pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// The environment variable [`OpenAiCompatClient::openai`] reads the key from.
pub const OPENAI_API_KEY_VAR: &str = "OPENAI_API_KEY";

/// A client for the OpenAI API, or any server that implements it: vLLM, LM
/// Studio, llama.cpp's server, OpenRouter, Groq, or Ollama's `/v1` endpoint.
///
/// # Example
///
/// ```
/// use lvv::backend::{ChatBackend, OpenAiCompatClient};
///
/// // A local vLLM server, which needs no key.
/// let client = OpenAiCompatClient::new("http://localhost:8000/v1", "qwen2.5-7b", None);
/// assert_eq!(client.identity().kind, "openai");
/// ```
#[derive(Debug, Clone)]
pub struct OpenAiCompatClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    timeout: Duration,
}

impl OpenAiCompatClient {
    /// A client for `model` on OpenAI's API.
    ///
    /// The key comes from `OPENAI_API_KEY`. When that variable isn't set, a
    /// `.env` file in the current directory or one of its parents is loaded
    /// first, if there is one.
    ///
    /// # Errors
    ///
    /// Returns a fatal [`BackendError`] when no key is found.
    pub fn openai(model: impl Into<String>) -> Result<Self, BackendError> {
        let api_key = resolve_api_key(
            || std::env::var(OPENAI_API_KEY_VAR).ok(),
            || {
                // A missing `.env` file is fine: the key may come from elsewhere.
                let _ = dotenvy::dotenv();
            },
        )?;
        Ok(Self::new(OPENAI_BASE_URL, model, Some(api_key)))
    }

    /// A client for `model` at `base_url`, such as `http://localhost:8000/v1`.
    /// `api_key` is sent as a bearer token when set.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        OpenAiCompatClient {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            api_key,
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
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        });
        if let Some(schema) = &request.schema {
            // `strict: true` demands schemas that `schemars` doesn't guarantee
            // (every property required, no additional properties). The reply is
            // parsed and checked locally either way.
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": schema.name,
                    "schema": schema.schema,
                    "strict": false,
                },
            });
        }
        body
    }
}

/// Reads the key, loading a `.env` file only when the first lookup fails.
fn resolve_api_key(
    lookup: impl Fn() -> Option<String>,
    load_dotenv: impl FnOnce(),
) -> Result<String, BackendError> {
    if let Some(key) = lookup().filter(|k| !k.is_empty()) {
        return Ok(key);
    }
    load_dotenv();
    lookup().filter(|k| !k.is_empty()).ok_or_else(|| {
        BackendError::config(format!(
            "{OPENAI_API_KEY_VAR} is not set, in the environment or in a .env file"
        ))
    })
}

#[derive(Deserialize)]
struct ChatReply {
    choices: Vec<Choice>,
    usage: Option<ReplyUsage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ReplyUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Deserialize)]
struct EmbedReply {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
    index: usize,
}

#[async_trait]
impl ChatBackend for OpenAiCompatClient {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            kind: "openai".into(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
        }
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BackendError> {
        let url = join_url(&self.base_url, "/chat/completions");
        let reply: ChatReply = post_json(
            &self.client,
            &url,
            self.api_key.as_deref(),
            self.timeout,
            &self.chat_body(request),
        )
        .await?;
        let text = reply
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .unwrap_or_default();
        Ok(ChatResponse {
            text,
            usage: reply.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
            }),
        })
    }
}

#[async_trait]
impl EmbedBackend for OpenAiCompatClient {
    fn identity(&self) -> BackendIdentity {
        ChatBackend::identity(self)
    }

    async fn embed(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, BackendError> {
        let url = join_url(&self.base_url, "/embeddings");
        let body = json!({"model": self.model, "input": inputs});
        let mut reply: EmbedReply = post_json(
            &self.client,
            &url,
            self.api_key.as_deref(),
            self.timeout,
            &body,
        )
        .await?;
        reply.data.sort_by_key(|item| item.index);
        Ok(reply.data.into_iter().map(|item| item.embedding).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::backend::{ErrorKind, OutputSchema, stub};

    #[test]
    fn key_from_the_environment_skips_dotenv() {
        let loaded = Cell::new(false);
        let key = resolve_api_key(|| Some("sk-env".into()), || loaded.set(true)).unwrap();
        assert_eq!(key, "sk-env");
        assert!(!loaded.get(), "a .env file must not be required");
    }

    #[test]
    fn key_from_dotenv_when_the_environment_has_none() {
        let loaded = Cell::new(false);
        let key = resolve_api_key(
            || loaded.get().then(|| "sk-dotenv".to_string()),
            || loaded.set(true),
        )
        .unwrap();
        assert_eq!(key, "sk-dotenv");
    }

    #[test]
    fn missing_key_names_the_variable() {
        let err = resolve_api_key(|| None, || {}).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Fatal);
        assert!(err.message.contains(OPENAI_API_KEY_VAR));
    }

    #[test]
    fn chat_body_has_exactly_the_system_and_user_messages() {
        let client = OpenAiCompatClient::new("http://x/v1", "gpt-4o-mini", None);
        let body = client.chat_body(&ChatRequest {
            system: "Extract keywords".into(),
            user: "text".into(),
            schema: None,
        });
        assert_eq!(
            body["messages"],
            json!([
                {"role": "system", "content": "Extract keywords"},
                {"role": "user", "content": "text"},
            ])
        );
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn chat_body_sends_the_schema_as_response_format() {
        let client = OpenAiCompatClient::new("http://x/v1", "gpt-4o-mini", None);
        let body = client.chat_body(&ChatRequest {
            system: "s".into(),
            user: "u".into(),
            schema: Some(OutputSchema {
                name: "Tags".into(),
                schema: json!({"type": "object"}),
            }),
        });
        assert_eq!(body["response_format"]["type"], json!("json_schema"));
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            json!("Tags")
        );
        assert_eq!(
            body["response_format"]["json_schema"]["schema"],
            json!({"type": "object"})
        );
    }

    #[tokio::test]
    async fn chat_sends_the_key_and_parses_the_reply() {
        let server = stub::StubServer::start(vec![stub::StubResponse::ok(
            r#"{"choices":[{"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}"#,
        )])
        .await;
        let client = OpenAiCompatClient::new(server.url("/v1"), "m", Some("sk-test".into()));
        let response = client
            .chat(&ChatRequest {
                system: "s".into(),
                user: "u".into(),
                schema: None,
            })
            .await
            .unwrap();
        assert_eq!(response.text, "hi");
        assert_eq!(response.usage.unwrap().total(), 6);
        let sent = server.requests();
        assert_eq!(sent[0].path, "/v1/chat/completions");
        assert_eq!(sent[0].authorization.as_deref(), Some("Bearer sk-test"));
    }

    #[tokio::test]
    async fn embeddings_come_back_in_input_order() {
        let server = stub::StubServer::start(vec![stub::StubResponse::ok(
            r#"{"data":[{"embedding":[2.0],"index":1},{"embedding":[1.0],"index":0}]}"#,
        )])
        .await;
        let client = OpenAiCompatClient::new(server.url("/v1"), "m", None);
        let vectors = client.embed(&["a", "b"]).await.unwrap();
        assert_eq!(vectors, vec![vec![1.0], vec![2.0]]);
        assert_eq!(server.requests()[0].body["input"], json!(["a", "b"]));
    }
}
