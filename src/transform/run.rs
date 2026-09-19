use std::{
    future::{Future, IntoFuture},
    path::PathBuf,
    pin::Pin,
    time::Duration,
};

use futures::StreamExt;

use super::{
    cache::CompletionCache,
    definition::Transform,
    error::{CompletionError, FatalCause, InputError, RunError},
    report::{Outcome, Progress, Report},
};
use crate::backend::{BackendError, ChatBackend, ChatRequest, ChatResponse, ErrorKind};

/// The longest a retry waits, whatever the backoff or `Retry-After` say.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

type ProgressFn<'a> = dyn FnMut(&Progress) + Send + 'a;

pub(crate) enum CacheSource {
    Path(PathBuf),
    #[cfg(test)]
    Ready(CompletionCache),
}

pub(crate) struct Options<'a> {
    concurrency: usize,
    retries: u32,
    backoff: Duration,
    circuit_breaker: usize,
    cache: Option<CacheSource>,
    on_progress: Option<Box<ProgressFn<'a>>>,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Options {
            concurrency: 1,
            retries: 2,
            backoff: Duration::from_millis(500),
            circuit_breaker: 3,
            cache: None,
            on_progress: None,
        }
    }
}

/// The run options shared by [`Run`] and [`Complete`].
macro_rules! run_options {
    () => {
        /// How many requests may be in flight at once. Defaults to 1.
        ///
        /// Outputs are still applied in a single loop, so the result doesn't
        /// depend on the order replies arrive in.
        pub fn concurrency(mut self, requests: usize) -> Self {
            self.options.concurrency = requests.max(1);
            self
        }

        /// How many times a request that fails with a retryable error is
        /// sent again. Defaults to 2. Fatal and record-specific failures are
        /// never retried.
        pub fn retries(mut self, retries: u32) -> Self {
            self.options.retries = retries;
            self
        }

        /// The wait before the first retry, doubled for each one after, plus
        /// up to 25% jitter, and capped at 30 seconds. A `Retry-After` from
        /// the server takes precedence. Defaults to 500 ms.
        pub fn retry_backoff(mut self, backoff: Duration) -> Self {
            self.options.backoff = backoff;
            self
        }

        /// Stops the run when this many requests in a row fail with fatal
        /// errors before the backend has answered once, as happens with a
        /// wrong model name or API key. Defaults to 3; 0 turns it off.
        pub fn circuit_breaker(mut self, threshold: usize) -> Self {
            self.options.circuit_breaker = threshold;
            self
        }

        /// Keeps completed outputs in the JSON Lines file at `path` and
        /// reuses them instead of calling the model again.
        ///
        /// An output is reused only for the same backend, model, prompt,
        /// schema and input, so rerunning an interrupted run sends only the
        /// records that didn't finish, and changing the prompt sends them all.
        pub fn cache(mut self, path: impl Into<PathBuf>) -> Self {
            self.options.cache = Some(CacheSource::Path(path.into()));
            self
        }

        /// Calls `hook` after each record gets an outcome.
        ///
        /// The run itself never prints. It also emits `tracing` events: one
        /// at `debug` level per record and a summary at `info` level.
        pub fn on_progress(mut self, hook: impl FnMut(&Progress) + Send + 'a) -> Self {
            self.options.on_progress = Some(Box::new(hook));
            self
        }
    };
}

/// A pending [`Llm::run`](super::Llm::run). Set options, then `.await` it.
#[must_use = "a run does nothing until it is awaited"]
pub struct Run<'a, T, O> {
    pub(crate) backend: &'a dyn ChatBackend,
    pub(crate) transform: &'a Transform<T, O>,
    pub(crate) records: &'a mut [T],
    pub(crate) options: Options<'a>,
}

impl<'a, T, O> Run<'a, T, O> {
    run_options!();

    #[cfg(test)]
    pub(crate) fn cache_store(mut self, cache: CompletionCache) -> Self {
        self.options.cache = Some(CacheSource::Ready(cache));
        self
    }
}

/// A pending [`Llm::complete`](super::Llm::complete). Set options, then
/// `.await` it.
#[must_use = "a run does nothing until it is awaited"]
pub struct Complete<'a, T, O> {
    pub(crate) backend: &'a dyn ChatBackend,
    pub(crate) transform: &'a Transform<T, O>,
    pub(crate) records: &'a [T],
    pub(crate) options: Options<'a>,
}

impl<'a, T, O> Complete<'a, T, O> {
    run_options!();
}

impl<'a, T, O> IntoFuture for Run<'a, T, O>
where
    T: Send + Sync + 'static,
    O: Send + 'static,
{
    type Output = Result<Report, RunError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let Run {
                backend,
                transform,
                records,
                options,
            } = self;
            let apply = transform.apply_fn().ok_or_else(|| RunError::Fatal {
                completed: 0,
                source: FatalCause::Config(
                    "the transform has no apply step; add one with Transform::apply, \
                     or use Llm::complete to get the outputs instead"
                        .into(),
                ),
            })?;
            let inputs = records.iter().map(|r| transform.render(r)).collect();
            let mut outcomes: Vec<Option<Outcome>> = records.iter().map(|_| None).collect();
            let tokens = execute(backend, transform, inputs, options, |index, delivery| {
                outcomes[index] = Some(match delivery {
                    Delivery::Fresh(output) => {
                        apply(&mut records[index], output);
                        Outcome::Applied
                    }
                    Delivery::Cached(output) => {
                        apply(&mut records[index], output);
                        Outcome::Cached
                    }
                    Delivery::Failed(err) => Outcome::Failed(err),
                });
            })
            .await?;
            Ok(Report {
                outcomes: outcomes
                    .into_iter()
                    .map(|o| o.expect("a finished run has an outcome for every record"))
                    .collect(),
                tokens,
            })
        })
    }
}

impl<'a, T, O> IntoFuture for Complete<'a, T, O>
where
    T: Send + Sync + 'static,
    O: Send + 'static,
{
    type Output = Result<Vec<Result<O, CompletionError>>, RunError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let Complete {
                backend,
                transform,
                records,
                options,
            } = self;
            let inputs = records.iter().map(|r| transform.render(r)).collect();
            let mut results: Vec<Option<Result<O, CompletionError>>> =
                records.iter().map(|_| None).collect();
            execute(backend, transform, inputs, options, |index, delivery| {
                results[index] = Some(match delivery {
                    Delivery::Fresh(output) | Delivery::Cached(output) => Ok(output),
                    Delivery::Failed(err) => Err(err),
                });
            })
            .await?;
            Ok(results
                .into_iter()
                .map(|r| r.expect("a finished run has a result for every record"))
                .collect())
        })
    }
}

pub(crate) enum Delivery<O> {
    Fresh(O),
    Cached(O),
    Failed(CompletionError),
}

/// Sends every record that isn't cached, hands each outcome to `deliver` as
/// it arrives, and returns the tokens used.
async fn execute<T, O>(
    backend: &dyn ChatBackend,
    transform: &Transform<T, O>,
    inputs: Vec<Result<String, InputError>>,
    mut options: Options<'_>,
    mut deliver: impl FnMut(usize, Delivery<O>),
) -> Result<u64, RunError> {
    let spec = &transform.output;
    let identity = backend.identity();
    let mut progress = Progress {
        total: inputs.len(),
        ..Progress::default()
    };
    let report = |progress: &Progress, hook: &mut Option<Box<ProgressFn<'_>>>| {
        if let Some(hook) = hook {
            hook(progress);
        }
    };

    let mut cache = match options.cache.take() {
        None => None,
        Some(CacheSource::Path(path)) => Some(CompletionCache::open(&path).await.map_err(
            |err| RunError::Fatal {
                completed: 0,
                source: FatalCause::Cache(err),
            },
        )?),
        #[cfg(test)]
        Some(CacheSource::Ready(cache)) => Some(cache),
    };

    // Resolve what doesn't need the model: unrenderable inputs and cache hits.
    let mut pending = Vec::new();
    for (index, input) in inputs.into_iter().enumerate() {
        let input = match input {
            Ok(input) => input,
            Err(err) => {
                progress.failed += 1;
                progress.done += 1;
                deliver(index, Delivery::Failed(err.into()));
                report(&progress, &mut options.on_progress);
                continue;
            }
        };
        let key = cache.as_ref().map(|_| {
            CompletionCache::key(&identity, transform.prompt(), spec.schema.as_ref(), &input)
        });
        let hit = cache
            .as_ref()
            .zip(key.as_ref())
            .and_then(|(cache, key)| cache.get(key))
            .and_then(|value| (spec.decode)(value.clone()));
        match hit {
            Some(output) => {
                progress.cached += 1;
                progress.done += 1;
                deliver(index, Delivery::Cached(output));
                report(&progress, &mut options.on_progress);
            }
            None => pending.push((index, input, key)),
        }
    }

    let (retries, backoff) = (options.retries, options.backoff);
    let requests = pending.into_iter().map(|(index, input, key)| {
        let request = ChatRequest {
            system: transform.prompt().to_string(),
            user: input,
            schema: spec.schema.clone(),
        };
        async move {
            let result = call_with_retry(backend, &request, retries, backoff).await;
            (index, key, result)
        }
    });
    let mut replies = futures::stream::iter(requests).buffer_unordered(options.concurrency);

    let mut backend_answered = false;
    let mut fatal_streak = 0;
    while let Some((index, key, result)) = replies.next().await {
        let completed = progress.succeeded + progress.cached;
        let delivery = match result {
            Ok(ChatResponse { text, usage }) => {
                backend_answered = true;
                progress.tokens += usage.map_or(0, |u| u.total());
                match (spec.parse)(&text) {
                    Ok(output) => {
                        if let (Some(cache), Some(key), Some(value)) =
                            (cache.as_mut(), key, (spec.encode)(&output))
                        {
                            cache
                                .insert(key, value)
                                .await
                                .map_err(|err| RunError::Fatal {
                                    completed,
                                    source: FatalCause::Cache(err),
                                })?;
                        }
                        progress.succeeded += 1;
                        Delivery::Fresh(output)
                    }
                    Err(err) => {
                        progress.failed += 1;
                        Delivery::Failed(err)
                    }
                }
            }
            Err(err) => {
                if err.kind == ErrorKind::Fatal && !backend_answered && options.circuit_breaker > 0
                {
                    fatal_streak += 1;
                    if fatal_streak >= options.circuit_breaker {
                        tracing::warn!(error = %err, "circuit breaker open, stopping the run");
                        return Err(RunError::Fatal {
                            completed,
                            source: FatalCause::CircuitOpen {
                                failures: fatal_streak,
                                last: err,
                            },
                        });
                    }
                }
                progress.failed += 1;
                Delivery::Failed(CompletionError::Backend(err))
            }
        };
        if let Delivery::Failed(err) = &delivery {
            tracing::debug!(index, error = %err, "record failed");
        } else {
            tracing::debug!(index, "record completed");
        }
        progress.done += 1;
        deliver(index, delivery);
        report(&progress, &mut options.on_progress);
    }

    tracing::info!(
        total = progress.total,
        succeeded = progress.succeeded,
        cached = progress.cached,
        failed = progress.failed,
        tokens = progress.tokens,
        "transform run finished"
    );
    Ok(progress.tokens)
}

async fn call_with_retry(
    backend: &dyn ChatBackend,
    request: &ChatRequest,
    retries: u32,
    backoff: Duration,
) -> Result<ChatResponse, BackendError> {
    let mut attempt = 0;
    loop {
        match backend.chat(request).await {
            Err(err) if err.is_retryable() && attempt < retries => {
                let wait = err
                    .retry_after
                    .unwrap_or_else(|| backoff.saturating_mul(2u32.saturating_pow(attempt)))
                    .min(MAX_BACKOFF);
                let jitter_ms = (wait.as_millis() as u64) / 4;
                let jitter = if jitter_ms == 0 {
                    Duration::ZERO
                } else {
                    Duration::from_millis(rand::random_range(0..=jitter_ms))
                };
                attempt += 1;
                tracing::debug!(attempt, error = %err, ?wait, "retrying request");
                tokio::time::sleep(wait + jitter).await;
            }
            result => return result,
        }
    }
}
