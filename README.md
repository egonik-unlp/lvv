# lvv

`lvv` is a Rust library for turning structured datasets into vector embeddings
and loading them into one or more storage backends. It supports Ollama and
OpenAI models, reusable embedding caches, sequential job queues, Qdrant, flat
files, optional SQL, HTTP and PostgreSQL connectors, and derive macros that let
your own Rust types describe what gets embedded and what gets stored. Records
can also be transformed by an LLM, for example summarized, before they are
embedded.

## Pipeline

1. Load records with a file, SQL, HTTP or PostgreSQL source, or create a
   `DataSet` directly.
2. Optionally transform the records with an LLM; see
   [Transforming records with LLMs](#transforming-records-with-llms).
3. Configure an embedding `Job` with an Ollama or OpenAI model.
4. Add the job to a `JobQueue`.
5. Register one or more sinks and run the queue.

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
is a complete lvv pipeline over derived structs. It reads records from JSON
Lines, CSV and JSON files with `FileSource`, turns them into points with the
derives, embeds the descriptions with Ollama using a `Cache`, and runs a
`JobQueue` that writes one Qdrant collection per category.

```sh
cargo run --example derive --features derive                  # load and print the points
cargo run --example derive --features derive -- --embed       # embed and build the jobs
cargo run --example derive --features derive -- --embed --qdrant http://localhost:6334
```

## Transforming records with LLMs

`inference::CompletionModel` sends records to an Ollama or OpenAI chat model
before they are embedded, to summarize long text, extract keywords, translate,
or clean up inconsistent fields. The prompt you give it is the system prompt,
and each record is sent as JSON in its own chat.

1. Create a `CompletionModel` with a model name and your instructions.
2. Run it over the records:
   - `perform_completion` returns the response texts, in record order.
   - `perform_completion_dump_inelegant` stores each response in its record
     through `FieldEnhanceable`, and saves the updated records to a JSON file
     as it goes.
3. Embed the result: build a `DataSet` from the updated records or, with the
   `derive` feature, mark the field that holds the response
   `#[lvv(description)]`.

```rust,ignore
use lvv::inference::{CompletionModel, completion_model::FieldEnhanceable};
use lvv::transform::transform::VectorDatabaseItem;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, VectorDatabaseItem)]
struct Article {
    #[lvv(description)]
    title: String,
    body: String,
    // Filled in by the model, then embedded with the title.
    #[lvv(description)]
    summary: Option<String>,
}

impl FieldEnhanceable for Article {
    fn set_field(&mut self, response: String) {
        self.summary = Some(response);
    }
}

let model = CompletionModel::new(
    "llama3.2",
    "Summarize the article in one sentence. Reply with the sentence only.",
)?;
model
    .perform_completion_dump_inelegant(articles, "articles.json".into())
    .await?;

let summarized: Vec<Article> =
    serde_json::from_str(&std::fs::read_to_string("articles.json")?)?;
let drafts = summarized
    .iter()
    .map(Article::try_into_database_item)
    .collect::<anyhow::Result<Vec<_>>>()?;
// drafts[0].description is "<title>\n<summary>".
```

Things to know:

- Records are sent one at a time.
- Records that fail are left out of the results instead of returning an error;
  a wrong model name returns no responses at all. Compare the number of
  responses with the number of records before pairing them.
- `perform_completion_dump_inelegant` returns only the response texts. Read the
  updated records from its file.
- Every chat also contains two fixed messages that introduce the record as
  input for summarization, whatever your prompt asks for.
- `perform_completion_and_live_dump` is unfinished and panics.

See the
[`CompletionModel` documentation](https://docs.rs/lvv/latest/lvv/inference/completion_model/struct.CompletionModel.html)
for details.

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
- `OPENAI_API_KEY` authenticates OpenAI requests, for both completions and
  embeddings.
- `QDRANT_API_KEY` authenticates remote Qdrant requests.

lvv loads a `.env` file before reading `OPENAI_API_KEY` or `QDRANT_API_KEY`,
and fails if there isn't one.

See the [API documentation](https://docs.rs/lvv) for detailed descriptions and
examples for each public API.

## License

Licensed under the Apache License, Version 2.0.
