# lvv

`lvv` is a Rust library for turning structured datasets into vector embeddings
and loading them into one or more storage backends. It supports Ollama and
OpenAI models, reusable embedding caches, sequential job queues, Qdrant, flat
files, and optional PostgreSQL source and sink connectors.

## Pipeline

1. Load records with a file or PostgreSQL source, or create a `DataSet` directly.
2. Configure an embedding `Job` with an Ollama or OpenAI model.
3. Add the job to a `JobQueue`.
4. Register one or more sinks and run the queue.

```rust,no_run
use lvv::{
    db::{Distance, QdrantSink},
    db::vector_database::{DatabaseParams, Location},
    intake::dataset::DataSet,
    jobs::{JobBuilder, Provider},
    jobs::job_queue::JobQueue,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dataset = DataSet::new(
        "articles.json",
        "articles",
        vec![serde_json::json!({"title": "Rust documentation"})],
    );

    let job = JobBuilder::default()
        .dataset(dataset)
        .provider(Provider::Ollama("nomic-embed-text".into()))
        .dims(768_u64)
        .extends(false)
        .distance(Distance::Cosine)
        .collection_name()
        .build()?;

    let params = DatabaseParams::new(
        Location::new_local("http://localhost:6334"),
        "articles".into(),
        Distance::Cosine,
        768,
    );
    let mut queue = JobQueue::from_vec(vec![job]);
    queue.with_sink(Arc::new(QdrantSink::new(params)));
    queue.run().await
}
```

## Configuration

- `OLLAMA_URL` selects the Ollama endpoint. It defaults to
  `http://127.0.0.1:11434`.
- `OPENAI_API_KEY` authenticates OpenAI requests.
- `QDRANT_API_KEY` authenticates remote Qdrant requests.
- The `postgres` Cargo feature enables `PostgresSource` and `PostgresSink`.

```toml
[dependencies]
lvv = { version = "0.4.4", features = ["postgres"] }
```

See the [API documentation](https://docs.rs/lvv) for detailed descriptions and
examples for each public API.

## License

Licensed under the Apache License, Version 2.0.
