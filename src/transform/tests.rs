//! One test per scenario of `specs/llm-transform/spec.md`, against a mock backend.

use std::{
    io,
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWrite;

use super::{
    CompletionError, ErrorClass, FatalCause, Llm, Outcome, Progress, RunError, Transform,
    cache::CompletionCache,
};
use crate::backend::mock::{MockChat, http_error, timeout};

#[derive(Debug, Clone, Serialize, PartialEq)]
struct Position {
    title: String,
    summary: Option<String>,
    keywords: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Tags {
    keywords: Vec<String>,
}

fn positions(n: usize) -> Vec<Position> {
    (0..n)
        .map(|i| Position {
            title: format!("p{i}"),
            summary: None,
            keywords: Vec::new(),
        })
        .collect()
}

fn llm(mock: &Arc<MockChat>) -> Llm {
    Llm::from_backend(mock.clone())
}

fn summarize() -> Transform<Position> {
    Transform::text("Summarize the position.")
        .input(|p: &Position| p.title.clone())
        .apply(|p: &mut Position, summary| p.summary = Some(summary))
}

fn tag() -> Transform<Position, Tags> {
    Transform::structured("List keywords.")
        .input(|p: &Position| p.title.clone())
        .apply(|p: &mut Position, tags: Tags| p.keywords = tags.keywords)
}

fn temp_cache(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "lvv-transform-test-{}-{name}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    path
}

/// Accepts `writes` writes, then fails every write.
struct FailAfter {
    writes: usize,
}

impl AsyncWrite for FailAfter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.writes == 0 {
            return Poll::Ready(Err(io::Error::other("disk full")));
        }
        self.writes -= 1;
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

// Requirement: One outcome per record in input order

#[tokio::test]
async fn a_failure_in_the_middle_keeps_alignment() {
    let mock = Arc::new(MockChat::new(|req, _| {
        if req.user == "p1" {
            Err(http_error(400))
        } else {
            Ok(format!("summary of {}", req.user))
        }
    }));
    let records = positions(3);
    let results = llm(&mock).complete(&summarize(), &records).await.unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].as_ref().unwrap(), "summary of p0");
    assert!(results[1].is_err());
    assert_eq!(results[2].as_ref().unwrap(), "summary of p2");
}

#[tokio::test]
async fn run_applies_successes_and_reports_failures_by_index() {
    let mock = Arc::new(MockChat::new(|req, _| {
        if req.user == "p1" {
            Err(http_error(400))
        } else {
            Ok(format!("summary of {}", req.user))
        }
    }));
    let mut records = positions(3);
    let report = llm(&mock).run(&summarize(), &mut records).await.unwrap();
    assert_eq!(records[0].summary.as_deref(), Some("summary of p0"));
    assert_eq!(records[1].summary, None);
    assert_eq!(records[2].summary.as_deref(), Some("summary of p2"));
    let failures: Vec<_> = report.failures().collect();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].0, 1);
    assert!(matches!(failures[0].1, CompletionError::Backend(e) if e.status == Some(400)));
    assert!(report.ensure_all().is_err());
}

// Requirement: Transforms are values independent of the record type

#[tokio::test]
async fn two_transforms_fill_two_fields_of_the_same_type() {
    let mock = Arc::new(MockChat::new(|req, _| {
        Ok(if req.schema.is_some() {
            format!(r#"{{"keywords":["{}"]}}"#, req.user)
        } else {
            format!("summary of {}", req.user)
        })
    }));
    let llm = llm(&mock);
    let mut records = positions(2);
    llm.run(&summarize(), &mut records)
        .await
        .unwrap()
        .ensure_all()
        .unwrap();
    llm.run(&tag(), &mut records)
        .await
        .unwrap()
        .ensure_all()
        .unwrap();
    assert_eq!(records[1].summary.as_deref(), Some("summary of p1"));
    assert_eq!(records[1].keywords, vec!["p1".to_string()]);
}

#[tokio::test]
async fn custom_input_is_the_user_message() {
    let mock = Arc::new(MockChat::echo());
    let transform: Transform<Position> =
        Transform::text("s").input(|p: &Position| format!("title={}", p.title));
    llm(&mock)
        .complete(&transform, &positions(1))
        .await
        .unwrap();
    assert_eq!(mock.requests()[0].user, "title=p0");
}

#[tokio::test]
async fn the_default_input_is_the_record_as_json() {
    let mock = Arc::new(MockChat::echo());
    let transform: Transform<Position> = Transform::text("s");
    let records = positions(1);
    llm(&mock).complete(&transform, &records).await.unwrap();
    assert_eq!(
        mock.requests()[0].user,
        serde_json::to_string(&records[0]).unwrap()
    );
}

// Requirement: Only the caller's prompt reaches the model

#[tokio::test]
async fn only_the_system_prompt_and_the_input_are_sent() {
    let mock = Arc::new(MockChat::echo());
    let transform: Transform<Position> =
        Transform::text("Extract keywords").input(|p: &Position| p.title.clone());
    llm(&mock)
        .complete(&transform, &positions(1))
        .await
        .unwrap();
    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].system, "Extract keywords");
    assert_eq!(requests[0].user, "p0");
    assert_eq!(requests[0].schema, None);
}

// Requirement: Structured output

#[tokio::test]
async fn structured_output_is_parsed_and_the_schema_is_sent() {
    let mock = Arc::new(MockChat::new(|_, _| Ok(r#"{"keywords":["rust"]}"#.into())));
    let mut records = positions(1);
    llm(&mock)
        .run(&tag(), &mut records)
        .await
        .unwrap()
        .ensure_all()
        .unwrap();
    assert_eq!(records[0].keywords, vec!["rust".to_string()]);
    let schema = mock.requests()[0].schema.clone().expect("schema sent");
    assert_eq!(schema.name, "Tags");
    assert!(schema.schema["properties"]["keywords"].is_object());
}

#[tokio::test]
async fn an_unparseable_response_fails_only_that_record() {
    let mock = Arc::new(MockChat::new(|req, _| {
        Ok(if req.user == "p0" {
            "not json".into()
        } else {
            r#"{"keywords":["ok"]}"#.into()
        })
    }));
    let mut records = positions(3);
    let report = llm(&mock).run(&tag(), &mut records).await.unwrap();
    match &report.outcomes()[0] {
        Outcome::Failed(CompletionError::Parse { raw, .. }) => assert_eq!(raw, "not json"),
        other => panic!("expected a parse failure, got {other:?}"),
    }
    assert_eq!(report.counts().applied, 2);
    assert_eq!(records[2].keywords, vec!["ok".to_string()]);
}

// Requirement: Typed and classified errors

#[test]
fn errors_are_classified() {
    let class = |err: CompletionError| err.class();
    assert_eq!(class(http_error(429).into()), ErrorClass::Retryable);
    assert_eq!(class(http_error(503).into()), ErrorClass::Retryable);
    assert_eq!(class(timeout().into()), ErrorClass::Retryable);
    assert_eq!(class(http_error(401).into()), ErrorClass::Fatal);
    assert_eq!(class(http_error(404).into()), ErrorClass::Fatal);
    assert_eq!(class(CompletionError::EmptyResponse), ErrorClass::Record);
    let parse = serde_json::from_str::<Tags>("x").unwrap_err();
    assert_eq!(
        class(CompletionError::Parse {
            raw: "x".into(),
            source: parse
        }),
        ErrorClass::Record
    );
}

#[tokio::test]
async fn an_empty_reply_is_a_record_failure() {
    let mock = Arc::new(MockChat::new(|_, _| Ok("  \n".into())));
    let results = llm(&mock)
        .complete(&summarize(), &positions(1))
        .await
        .unwrap();
    assert!(matches!(results[0], Err(CompletionError::EmptyResponse)));
}

// Requirement: Retries on retryable failures only

#[tokio::test]
async fn a_transient_failure_recovers() {
    let mock = Arc::new(MockChat::new(|_, call| {
        if call == 0 {
            Err(http_error(503))
        } else {
            Ok("fine".into())
        }
    }));
    let mut records = positions(1);
    let report = llm(&mock)
        .run(&summarize(), &mut records)
        .retries(2)
        .retry_backoff(Duration::from_millis(1))
        .await
        .unwrap();
    report.ensure_all().unwrap();
    assert_eq!(mock.calls(), 2);
}

#[tokio::test]
async fn retries_stop_at_the_limit() {
    let mock = Arc::new(MockChat::new(|_, _| Err(timeout())));
    let results = llm(&mock)
        .complete(&summarize(), &positions(1))
        .retries(2)
        .retry_backoff(Duration::from_millis(1))
        .await
        .unwrap();
    assert!(results[0].is_err());
    assert_eq!(mock.calls(), 3);
}

#[tokio::test]
async fn a_fatal_failure_is_not_retried() {
    let mock = Arc::new(MockChat::new(|_, _| Err(http_error(401))));
    let results = llm(&mock)
        .complete(&summarize(), &positions(1))
        .retries(2)
        .retry_backoff(Duration::from_millis(1))
        .circuit_breaker(0)
        .await
        .unwrap();
    assert!(results[0].is_err());
    assert_eq!(mock.calls(), 1);
}

// Requirement: Circuit breaker on fatal failures

#[tokio::test]
async fn a_wrong_model_name_stops_the_run_early() {
    let mock = Arc::new(MockChat::new(|_, _| Err(http_error(404))));
    let mut records = positions(100);
    let err = llm(&mock)
        .run(&summarize(), &mut records)
        .circuit_breaker(3)
        .await
        .unwrap_err();
    match err {
        RunError::Fatal {
            completed: 0,
            source: FatalCause::CircuitOpen { failures: 3, .. },
        } => {}
        other => panic!("expected an open circuit, got {other:?}"),
    }
    assert!(mock.calls() <= 3, "sent {} requests", mock.calls());
}

#[tokio::test]
async fn record_failures_do_not_open_the_circuit() {
    let mock = Arc::new(MockChat::new(|_, call| {
        Ok(if call < 5 {
            "not json".into()
        } else {
            r#"{"keywords":[]}"#.into()
        })
    }));
    let mut records = positions(10);
    let report = llm(&mock)
        .run(&tag(), &mut records)
        .circuit_breaker(3)
        .await
        .unwrap();
    assert_eq!(mock.calls(), 10);
    let failed: Vec<usize> = report.failures().map(|(i, _)| i).collect();
    assert_eq!(failed, vec![0, 1, 2, 3, 4]);
}

#[tokio::test]
async fn a_success_disarms_the_circuit() {
    let mock = Arc::new(MockChat::new(|req, _| {
        if req.user == "p0" {
            Ok("fine".into())
        } else {
            Err(http_error(404))
        }
    }));
    let mut records = positions(10);
    let report = llm(&mock).run(&summarize(), &mut records).await.unwrap();
    assert_eq!(report.counts().failed, 9);
    assert_eq!(mock.calls(), 10);
}

// Requirement: Fatal run errors keep completed work

#[tokio::test]
async fn a_cache_write_failure_keeps_the_completed_records() {
    let mock = Arc::new(MockChat::new(|req, _| {
        Ok(format!("summary of {}", req.user))
    }));
    let mut records = positions(5);
    let err = llm(&mock)
        .run(&summarize(), &mut records)
        .cache_store(CompletionCache::with_writer(FailAfter { writes: 3 }))
        .await
        .unwrap_err();
    match err {
        RunError::Fatal {
            completed: 3,
            source: FatalCause::Cache(_),
        } => {}
        other => panic!("expected a cache failure after 3 records, got {other:?}"),
    }
    let summaries: Vec<_> = records.iter().map(|r| r.summary.is_some()).collect();
    assert_eq!(summaries, vec![true, true, true, false, false]);
}

#[tokio::test]
async fn run_without_apply_fails_before_sending_anything() {
    let mock = Arc::new(MockChat::echo());
    let transform: Transform<Position> = Transform::text("s");
    let err = llm(&mock)
        .run(&transform, &mut positions(2))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        RunError::Fatal {
            completed: 0,
            source: FatalCause::Config(_)
        }
    ));
    assert_eq!(mock.calls(), 0);
}

// Requirement: Content-keyed completion cache

#[tokio::test]
async fn an_interrupted_run_resumes_from_the_cache() {
    let path = temp_cache("resume");
    let mut records = positions(10);

    // The first run finishes six records before it stops.
    let first = Arc::new(MockChat::new(|req, _| {
        Ok(format!("summary of {}", req.user))
    }));
    llm(&first)
        .run(&summarize(), &mut records[..6])
        .cache(&path)
        .await
        .unwrap();

    let second = Arc::new(MockChat::new(|req, _| {
        Ok(format!("summary of {}", req.user))
    }));
    let mut fresh = positions(10);
    let report = llm(&second)
        .run(&summarize(), &mut fresh)
        .cache(&path)
        .await
        .unwrap();
    assert_eq!(second.calls(), 4);
    assert_eq!(report.counts().cached, 6);
    assert_eq!(report.counts().applied, 4);
    assert!(fresh.iter().all(|r| r.summary.is_some()));
    assert_eq!(fresh[3].summary.as_deref(), Some("summary of p3"));
    std::fs::remove_file(path).ok();
}

#[tokio::test]
async fn a_prompt_change_invalidates_the_cache() {
    let path = temp_cache("prompt");
    let mock = Arc::new(MockChat::echo());
    let llm = llm(&mock);
    let mut records = positions(4);
    llm.run(&summarize(), &mut records)
        .cache(&path)
        .await
        .unwrap();
    assert_eq!(mock.calls(), 4);

    let reworded = Transform::text("Summarize the position briefly.")
        .input(|p: &Position| p.title.clone())
        .apply(|p: &mut Position, s| p.summary = Some(s));
    llm.run(&reworded, &mut records).cache(&path).await.unwrap();
    assert_eq!(mock.calls(), 8);
    std::fs::remove_file(path).ok();
}

#[tokio::test]
async fn a_different_model_does_not_reuse_the_cache() {
    let path = temp_cache("model");
    let a = Arc::new(MockChat::echo().with_model("a"));
    let b = Arc::new(MockChat::echo().with_model("b"));
    let mut records = positions(2);
    llm(&a)
        .run(&summarize(), &mut records)
        .cache(&path)
        .await
        .unwrap();
    llm(&b)
        .run(&summarize(), &mut records)
        .cache(&path)
        .await
        .unwrap();
    assert_eq!(b.calls(), 2);
    std::fs::remove_file(path).ok();
}

#[tokio::test]
async fn structured_outputs_are_cached_too() {
    let path = temp_cache("structured");
    let mock = Arc::new(MockChat::new(|_, _| Ok(r#"{"keywords":["rust"]}"#.into())));
    let llm = llm(&mock);
    let mut records = positions(2);
    llm.run(&tag(), &mut records).cache(&path).await.unwrap();
    let mut again = positions(2);
    let report = llm.run(&tag(), &mut again).cache(&path).await.unwrap();
    assert_eq!(mock.calls(), 2);
    assert_eq!(report.counts().cached, 2);
    assert_eq!(again[1].keywords, vec!["rust".to_string()]);
    std::fs::remove_file(path).ok();
}

#[tokio::test]
async fn a_truncated_cache_line_is_a_miss() {
    let path = temp_cache("truncated");
    let mock = Arc::new(MockChat::echo());
    let llm = llm(&mock);
    let mut records = positions(2);
    llm.run(&summarize(), &mut records)
        .cache(&path)
        .await
        .unwrap();
    // Cut the last entry off mid-write.
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, &text[..text.len() - 10]).unwrap();

    let mut again = positions(2);
    let report = llm
        .run(&summarize(), &mut again)
        .cache(&path)
        .await
        .unwrap();
    assert_eq!(report.counts().cached, 1);
    assert_eq!(report.counts().applied, 1);
    assert_eq!(mock.calls(), 3);
    std::fs::remove_file(path).ok();
}

// Requirement: Opt-in concurrency

#[tokio::test]
async fn a_concurrent_run_stays_aligned() {
    // Later records answer first.
    let mock = Arc::new(MockChat::echo().with_delay(|req| {
        let n: u64 = req.user[1..].parse().unwrap();
        Duration::from_millis(40 - 2 * n)
    }));
    let mut records = positions(20);
    let report = llm(&mock)
        .run(&summarize(), &mut records)
        .concurrency(4)
        .await
        .unwrap();
    report.ensure_all().unwrap();
    assert!(
        records
            .iter()
            .all(|r| r.summary.as_deref() == Some(r.title.as_str()))
    );
}

#[tokio::test]
async fn concurrency_runs_requests_in_parallel() {
    let in_flight = Arc::new(Mutex::new((0usize, 0usize)));
    let tracker = in_flight.clone();
    let mock = Arc::new(MockChat::echo().with_delay(move |_| {
        let mut state = tracker.lock().unwrap();
        state.0 += 1;
        state.1 = state.1.max(state.0);
        Duration::from_millis(20)
    }));
    // The delay closure only counts starts; with 8 records and concurrency 4,
    // the first 4 start before any finishes.
    llm(&mock)
        .complete(&summarize(), &positions(8))
        .concurrency(4)
        .await
        .unwrap();
    assert!(in_flight.lock().unwrap().1 >= 4);
}

// Requirement: No output to stdout

#[tokio::test]
async fn progress_is_reported_through_the_hook() {
    let mock = Arc::new(MockChat::echo());
    let seen = Arc::new(Mutex::new(Vec::<Progress>::new()));
    let sink = seen.clone();
    let report = llm(&mock)
        .run(&summarize(), &mut positions(5))
        .on_progress(move |p| sink.lock().unwrap().push(*p))
        .await
        .unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 5);
    let last = seen.last().unwrap();
    assert_eq!((last.total, last.done, last.succeeded), (5, 5, 5));
    assert_eq!(last.tokens, 10);
    assert_eq!(report.tokens(), 10);
}
