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

For every stage in one program, with the derive macros and an LLM transform,
see the [full example](#full-example) at the end of this README.

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

## Full example

[`examples/full_pipeline.rs`](https://github.com/egonik-unlp/lvv/blob/main/examples/full_pipeline.rs)
puts every stage together, with the `derive` feature:

1. `FileSource` loads positions, skills and projects from JSON Lines, CSV and
   JSON files into structs that derive `VectorDatabaseItem`.
2. `CompletionModel` writes a summary into each position, in a field marked
   `#[lvv(description)]`.
3. `#[derive(VectorDatabase)]` turns the records into points.
4. `EmbeddingProvider` embeds each category's descriptions, reusing vectors
   from a `Cache`.
5. Each category becomes a `Job` with precomputed embeddings, because a
   `JobQueue` would otherwise embed each row's JSON instead of its description.
6. A `JobQueue` writes the jobs to Qdrant, one collection per category.

It needs Ollama with a chat model and an embedding model: `llama3.2` and
`embeddinggemma` by default, or the models named by `CHAT_MODEL` and
`EMBEDDING_MODEL`. Set `QDRANT_URL` to write the points; without it, the
program stops after building the jobs. The data files are in
[`examples/data`](https://github.com/egonik-unlp/lvv/tree/main/examples/data).

```sh
cargo run --example full_pipeline --features derive
QDRANT_URL=http://localhost:6334 cargo run --example full_pipeline --features derive
```

```rust
use std::{collections::BTreeMap, path::Path};

use anyhow::Context;
use lvv::{
    cache::cache_embeddings::Cache,
    db::{
        Distance,
        vector_database::{DatabaseParams, Location},
    },
    inference::{CompletionModel, EmbeddingProvider},
    intake::{FileSource, Source, dataset::DataSet},
    jobs::{JobBuilder, Provider, job_queue::JobQueue},
    transform::transform::{VectorDatabase, VectorDatabaseItem, VectorPointDraft},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// A job, from `positions.jsonl`. The file's `salary` isn't a field, so it
/// never reaches the model or the payload.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
#[serde(rename_all = "camelCase")]
struct Position {
    /// Embedded, and stored under `role`.
    #[lvv(description, rename = "role")]
    job_title: String,
    organization: Option<String>,
    highlights: Vec<String>,
    started_on: String,
    /// Written by the LLM, then embedded with the title.
    #[lvv(description)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
}

/// A skill, from `skills.csv`.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
struct Skill {
    #[lvv(description)]
    name: String,
    #[lvv(description)]
    level: String,
    /// Neither embedded nor stored.
    #[lvv(skip)]
    internal_id: String,
}

/// A project, from `projects.json`.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
struct Project {
    #[lvv(description)]
    name: String,
    #[lvv(description)]
    summary: String,
    /// Stored under `stack`, but not embedded.
    #[lvv(rename = "stack")]
    technologies: Vec<String>,
}

/// Every record, one point each.
#[derive(VectorDatabase)]
struct Portfolio {
    positions: Vec<Position>,
    skills: Vec<Skill>,
    projects: Vec<Project>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let chat_model = env_or("CHAT_MODEL", "llama3.2");
    let embedding_model = env_or("EMBEDDING_MODEL", "embeddinggemma");

    // 1. Load: rows from three files, deserialized into the derived structs.
    let mut portfolio = Portfolio {
        positions: load("positions.jsonl").await?,
        skills: load("skills.csv").await?,
        projects: load("projects.json").await?,
    };

    // 2. Transform: the LLM writes a one-sentence summary of each position.
    let model = CompletionModel::new(
        chat_model.as_str(),
        "You write search summaries of job positions. The last user message is one \
         job position as JSON. Reply with a single sentence of at most 25 words that \
         describes the role, the organization and what the person did. Do not add \
         quotes, greetings or any other text.",
    )?;
    let summaries = model
        .perform_completion(portfolio.positions.iter().collect::<Vec<_>>())
        .await?;
    // Failed records are left out of the responses, so pairing responses with
    // records is only safe when every record got one.
    anyhow::ensure!(
        summaries.len() == portfolio.positions.len(),
        "{} of {} summaries failed; check that `{chat_model}` is a chat model \
         available in Ollama",
        portfolio.positions.len() - summaries.len(),
        portfolio.positions.len()
    );
    for (position, summary) in portfolio.positions.iter_mut().zip(summaries) {
        // Small models sometimes wrap the sentence in quotes anyway.
        position.summary = Some(summary.trim().trim_matches('"').to_string());
    }

    // 3. Derive: one point per record, grouped by category.
    let mut by_category: BTreeMap<String, Vec<VectorPointDraft>> = BTreeMap::new();
    for draft in portfolio.point_drafts()? {
        by_category
            .entry(draft.category.clone())
            .or_default()
            .push(draft);
    }

    // 4. Embed: each category's descriptions, reusing vectors from earlier runs.
    let cache_path = std::env::temp_dir().join("lvv-full-pipeline-cache.json");
    let cache_path = cache_path.to_str().context("cache path is not UTF-8")?;
    let mut cache = if Path::new(cache_path).exists() {
        Cache::from_json_file(cache_path)?
    } else {
        Cache::new()
    };
    let embedder = EmbeddingProvider::new(&embedding_model)?;

    let mut jobs = Vec::new();
    for (category, drafts) in &by_category {
        let descriptions: Vec<String> = drafts.iter().map(|d| d.description.clone()).collect();
        let embeddings = match cache.get_embedding(embedding_model.clone(), descriptions.clone()) {
            Some(cached) => cached.clone(),
            None => {
                let dataset = DataSet::new("portfolio", category.as_str(), descriptions.clone());
                let fresh = embedder.embed_properties(dataset).await?;
                cache.add_embedding(embedding_model.clone(), descriptions, fresh.clone());
                fresh
            }
        };

        // 5. Jobs: the payloads are the rows. The vectors are precomputed
        //    because a queue would embed each row's JSON, not the description.
        let rows = drafts
            .iter()
            .map(|d| serde_json::to_value(&d.payload))
            .collect::<Result<Vec<_>, _>>()?;
        let job = JobBuilder::default()
            .dataset(DataSet::new(
                "portfolio",
                format!("portfolio_{}", category.to_lowercase()),
                rows,
            ))
            .provider(Provider::Ollama(embedding_model.clone()))
            .dims(embeddings.first().map_or(0, Vec::len) as u64)
            // Points get random IDs, so leave a collection that already has
            // points untouched instead of adding duplicates on every run.
            .extends(false)
            .distance(Distance::Cosine)
            .embedding(embeddings)
            // Must come after the dataset, provider and distance.
            .collection_name()
            .build()?;
        println!("\n{} points -> `{}`", drafts.len(), job.collection_name);
        for draft in drafts {
            println!("  {:?}", draft.description);
        }
        jobs.push(job);
    }
    cache.to_json_file(cache_path)?;

    // 6. Write: the queue creates each job's collection in Qdrant.
    let Ok(url) = std::env::var("QDRANT_URL") else {
        println!("\nSet QDRANT_URL to write the points to Qdrant.");
        return Ok(());
    };
    // `Location::new_local` takes a `&'static str`.
    let url: &'static str = Box::leak(url.into_boxed_str());
    let dims = jobs.first().map_or(0, |job| job.dims) as u16;
    let mut queue = JobQueue::from_vec(jobs);
    queue.with_qdrant_sink(DatabaseParams::new(
        Location::new_local(url),
        "portfolio".into(),
        Distance::Cosine,
        dims,
    ));
    queue.run().await
}

/// Reads every row of `examples/data/<file>` with a `FileSource`, which picks
/// the format from the extension, and deserializes each row into `T`.
async fn load<T: DeserializeOwned>(file: &str) -> anyhow::Result<Vec<T>> {
    let path = format!("{}/examples/data/{file}", env!("CARGO_MANIFEST_DIR"));
    let mut records = Vec::new();
    for dataset in FileSource::new(&path, file)?.fetch().await? {
        for row in dataset.data.unwrap_or_default() {
            let record =
                serde_json::from_value(row).with_context(|| format!("parsing a row of {file}"))?;
            records.push(record);
        }
    }
    Ok(records)
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}
```

## License

Licensed under the Apache License, Version 2.0.
