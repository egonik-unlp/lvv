//! Scripted backends for unit tests.

use std::{
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;

use super::{
    BackendError, BackendIdentity, ChatBackend, ChatRequest, ChatResponse, EmbedBackend, ErrorKind,
    classify_status,
};

type ChatScript = dyn Fn(&ChatRequest, usize) -> Result<String, BackendError> + Send + Sync;
type DelayScript = dyn Fn(&ChatRequest) -> Duration + Send + Sync;

/// Answers each request with a closure of the request and the call number
/// (0-based), and records every request it receives.
pub(crate) struct MockChat {
    script: Box<ChatScript>,
    delay: Option<Box<DelayScript>>,
    calls: AtomicUsize,
    requests: Mutex<Vec<ChatRequest>>,
    model: String,
}

impl MockChat {
    pub(crate) fn new(
        script: impl Fn(&ChatRequest, usize) -> Result<String, BackendError> + Send + Sync + 'static,
    ) -> Self {
        MockChat {
            script: Box::new(script),
            delay: None,
            calls: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
            model: "mock-model".into(),
        }
    }

    /// Echoes the user message back.
    pub(crate) fn echo() -> Self {
        Self::new(|request, _| Ok(request.user.clone()))
    }

    pub(crate) fn with_delay(
        mut self,
        delay: impl Fn(&ChatRequest) -> Duration + Send + Sync + 'static,
    ) -> Self {
        self.delay = Some(Box::new(delay));
        self
    }

    pub(crate) fn with_model(mut self, model: &str) -> Self {
        self.model = model.into();
        self
    }

    pub(crate) fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub(crate) fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatBackend for MockChat {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            kind: "mock".into(),
            base_url: "mock://".into(),
            model: self.model.clone(),
        }
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BackendError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().unwrap().push(request.clone());
        if let Some(delay) = &self.delay {
            tokio::time::sleep(delay(request)).await;
        }
        (self.script)(request, call).map(|text| ChatResponse {
            text,
            usage: Some(super::Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
            }),
        })
    }
}

type EmbedScript = dyn Fn(&[&str]) -> Result<Vec<Vec<f32>>, BackendError> + Send + Sync;

/// Embeds with a closure and records every batch it receives.
pub(crate) struct MockEmbed {
    script: Box<EmbedScript>,
    batches: Mutex<Vec<Vec<String>>>,
}

impl MockEmbed {
    pub(crate) fn new(
        script: impl Fn(&[&str]) -> Result<Vec<Vec<f32>>, BackendError> + Send + Sync + 'static,
    ) -> Self {
        MockEmbed {
            script: Box::new(script),
            batches: Mutex::new(Vec::new()),
        }
    }

    /// One vector per input: `[input length]`.
    pub(crate) fn lengths() -> Self {
        Self::new(|inputs| Ok(inputs.iter().map(|s| vec![s.len() as f32]).collect()))
    }

    pub(crate) fn batches(&self) -> Vec<Vec<String>> {
        self.batches.lock().unwrap().clone()
    }
}

#[async_trait]
impl EmbedBackend for MockEmbed {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            kind: "mock".into(),
            base_url: "mock://".into(),
            model: "mock-embed".into(),
        }
    }

    async fn embed(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, BackendError> {
        self.batches
            .lock()
            .unwrap()
            .push(inputs.iter().map(|s| s.to_string()).collect());
        (self.script)(inputs)
    }
}

/// An error as a server with this status would produce it.
pub(crate) fn http_error(status: u16) -> BackendError {
    BackendError {
        kind: classify_status(status),
        status: Some(status),
        message: format!("mock status {status}"),
        retry_after: None,
    }
}

/// A retryable error without a status, like a timeout.
pub(crate) fn timeout() -> BackendError {
    BackendError {
        kind: ErrorKind::Retryable,
        status: None,
        message: "mock timeout".into(),
        retry_after: None,
    }
}
