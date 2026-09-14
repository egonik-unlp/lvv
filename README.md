# lvv

`lvv` is a Rust library for turning structured datasets into vector embeddings
and loading them into one or more storage backends. It supports Ollama and
OpenAI models, reusable embedding caches, sequential job queues, Qdrant, flat
files, optional SQL, HTTP and PostgreSQL connectors, and derive macros that let
your own Rust types describe what gets embedded and what gets stored.

## Pipeline

1. Load records with a file, SQL, HTTP or PostgreSQL source, or create a
   `DataSet` directly.
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

## Typed records with derive macros

The `derive` feature adds `#[derive(VectorDatabaseItem)]` and
`#[derive(VectorDatabase)]` from
[`lvv-macros`](https://crates.io/crates/lvv-macros). They are re-exported in
`lvv::transform::transform`, next to the traits they implement.

```toml
[dependencies]
lvv = { version = "0.5", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
```

```rust,ignore
use lvv::transform::transform::{VectorDatabase, VectorDatabaseItem};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, VectorDatabaseItem)]
#[serde(rename_all = "camelCase")]
struct Position {
    #[lvv(description, rename = "role")]
    job_title: String,
    #[lvv(description)]
    organization: Option<String>,
    started_on: String,
    #[lvv(skip)]
    internal_id: u64,
}

#[derive(VectorDatabase)]
struct Portfolio {
    positions: Vec<Position>,
    #[lvv(skip)]
    owner: String,
}

let drafts = portfolio.point_drafts()?;
```

Each item becomes a `VectorPointDraft`:

- `category`: the struct name, `"Position"`.
- `description`: the `#[lvv(description)]` fields joined with newlines, such
  as `"Researcher\nUNLP"`. This is the text you embed.
- `payload`: the struct serialized with serde, without `#[lvv(skip)]` fields,
  with `#[lvv(rename)]` applied and a `category` key added, such as
  `{"role": "Researcher", "organization": "UNLP", "startedOn": "2020-03-01", "category": "Position"}`.

`#[derive(VectorDatabase)]` collects the points of fields typed `T`, `Vec<T>`,
`BTreeMap<K, T>` or `HashMap<K, T>`, each optionally wrapped in `Option`. See the
[`lvv-macros` documentation](https://docs.rs/lvv-macros) for every attribute,
and the
[`transform::transform` module](https://docs.rs/lvv/latest/lvv/transform/transform/)
for embedding and storing drafts.

[`examples/derive.rs`](https://github.com/egonik-unlp/lvv/blob/main/examples/derive.rs)
is a complete program that uses both derives: it prints the points of a small
portfolio, embeds them with Ollama and stores them in Qdrant.

```sh
cargo run --example derive --features derive                  # print the points
cargo run --example derive --features derive -- --embed       # embed with Ollama
cargo run --example derive --features derive -- --embed --qdrant http://localhost:6334
```

## Cargo features

No features are enabled by default.

| Feature    | Enables |
|------------|---------|
| `derive`   | `VectorDatabaseItem` and `VectorDatabase` derive macros from `lvv-macros` |
| `postgres` | `PostgresSource` and `PostgresSink` |
| `sql`      | `SqlSource` for SQLite and MySQL |
| `http`     | `HttpSource` for paginated JSON APIs |

```toml
[dependencies]
lvv = { version = "0.5", features = ["derive", "postgres"] }
```

## Configuration

- `OLLAMA_URL` selects the Ollama endpoint. It defaults to
  `http://127.0.0.1:11434`.
- `OPENAI_API_KEY` authenticates OpenAI requests.
- `QDRANT_API_KEY` authenticates remote Qdrant requests.

See the [API documentation](https://docs.rs/lvv) for detailed descriptions and
examples for each public API.

## License

Licensed under the Apache License, Version 2.0.
