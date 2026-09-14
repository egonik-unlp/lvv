//! Turn typed records into vector points with the `lvv-macros` derives.
//!
//! `#[derive(VectorDatabaseItem)]` makes each record produce one point: the
//! text to embed, a category, and the payload stored next to the vector.
//! `#[derive(VectorDatabase)]` collects the points of every record a container
//! holds. This example prints the points of a small portfolio, then optionally
//! embeds them with Ollama and stores them in Qdrant.
//!
//! ```text
//! cargo run --example derive --features derive
//! cargo run --example derive --features derive -- --embed
//! cargo run --example derive --features derive -- --embed --qdrant http://localhost:6334
//! ```

use std::collections::BTreeMap;

use clap::Parser;
use lvv::{
    db::{
        Distance, QdrantSink, Sink, SinkContext,
        vector_database::{DatabaseParams, Location},
    },
    inference::EmbeddingProvider,
    intake::dataset::DataSet,
    transform::transform::{IntoDescriptionValue, VectorDatabase, VectorDatabaseItem},
};
use serde::{Deserialize, Serialize};

/// A job. A `VectorDatabaseItem` needs `Serialize`, `Deserialize` and at least
/// one `#[lvv(description)]` field.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
#[serde(rename_all = "camelCase")]
struct Position {
    /// Embedded, and stored under `role` instead of `jobTitle`.
    #[lvv(description, rename = "role")]
    job_title: String,
    /// Embedded when present; `None` adds nothing to the description.
    #[lvv(description)]
    organization: Option<String>,
    /// Embedded one line per highlight.
    #[lvv(description)]
    highlights: Vec<String>,
    /// Stored as `startedOn`, following `rename_all`, but not embedded.
    started_on: String,
    /// Neither embedded nor stored.
    #[lvv(skip)]
    salary: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Level {
    Familiar,
    Proficient,
    Expert,
}

/// Lets `Level` be a description field. `String`, numbers, `Option` and `Vec`
/// already implement `IntoDescriptionValue`.
impl IntoDescriptionValue for Level {
    fn into_description_value(&self) -> String {
        match self {
            Level::Familiar => "familiar",
            Level::Proficient => "proficient",
            Level::Expert => "expert",
        }
        .into()
    }
}

#[derive(Serialize, Deserialize, VectorDatabaseItem)]
struct Skill {
    #[lvv(description)]
    name: String,
    #[lvv(description)]
    level: Level,
}

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

/// Tuple structs work too.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
struct Interest(#[lvv(description)] String);

/// Holds the records. Each field yields points according to its type, and the
/// container itself needs no serde derives.
#[derive(VectorDatabase)]
struct Portfolio {
    /// One point, or none.
    current_position: Option<Position>,
    /// One point per element.
    past_positions: Vec<Position>,
    skills: Vec<Skill>,
    /// One point per value, in key order. Keys are not used.
    projects: BTreeMap<String, Project>,
    interests: Option<Vec<Interest>>,
    /// Not a record, so it must be skipped.
    #[lvv(skip)]
    owner: String,
}

#[derive(Parser)]
#[command(about = "Turn a portfolio into vector points with the lvv-macros derives")]
struct Args {
    /// Embed each point's description with Ollama.
    #[arg(long)]
    embed: bool,
    /// Ollama embedding model.
    #[arg(long, default_value = "embeddinggemma")]
    model: String,
    /// Store the points in Qdrant at this gRPC URL.
    #[arg(long, requires = "embed")]
    qdrant: Option<String>,
    /// Qdrant collection for the points.
    #[arg(long, default_value = "portfolio")]
    collection: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let portfolio = sample_portfolio();

    // One draft per record, in field order.
    let drafts = portfolio.point_drafts()?;
    println!("{} points for {}", drafts.len(), portfolio.owner);
    for draft in &drafts {
        println!("\n[{}]", draft.category);
        println!("  description: {:?}", draft.description);
        println!("  payload:     {}", serde_json::to_string(&draft.payload)?);
    }

    // A single record builds its own draft, for example to re-index one edit.
    let draft = portfolio.skills[0].try_into_database_item()?;
    println!("\nFirst skill on its own: {:?}", draft.description);

    if !args.embed {
        println!("\nDry run: pass --embed to embed the descriptions with Ollama.");
        return Ok(());
    }

    // Embeddings come back in the same order as the descriptions.
    let descriptions: Vec<String> = drafts.iter().map(|d| d.description.clone()).collect();
    let embeddings = EmbeddingProvider::new(&args.model)?
        .embed_properties(DataSet::new("portfolio", "points", descriptions))
        .await?;
    let dims = embeddings.first().map_or(0, Vec::len) as u64;
    println!(
        "\nEmbedded {} descriptions with {} ({dims} dimensions)",
        embeddings.len(),
        args.model
    );

    let Some(url) = args.qdrant else {
        println!("Pass --qdrant <url> to store the points.");
        return Ok(());
    };

    // Each payload is stored with its vector. All categories share one
    // collection; filter on the `category` payload key.
    let rows = drafts
        .iter()
        .map(|d| serde_json::to_value(&d.payload))
        .collect::<Result<Vec<_>, _>>()?;
    // `Location::new_local` takes a `&'static str`.
    let url: &'static str = Box::leak(url.into_boxed_str());
    let sink = QdrantSink::new(DatabaseParams::new(
        Location::new_local(url),
        args.collection.clone(),
        Distance::Cosine,
        dims as u16,
    ));
    sink.write(&SinkContext {
        collection_name: &args.collection,
        dims,
        // Points get random IDs, so leave a collection that already has points
        // untouched instead of adding duplicates on every run.
        extends: false,
        embeddings: &embeddings,
        rows: &rows,
    })
    .await?;
    println!(
        "Stored the points in Qdrant collection `{}` (skipped if it already had points)",
        args.collection
    );
    Ok(())
}

/// Ten points: one current position, two past positions, three skills, two
/// projects and two interests.
fn sample_portfolio() -> Portfolio {
    Portfolio {
        current_position: Some(Position {
            job_title: "Research software engineer".into(),
            organization: Some("Lensing Lab".into()),
            highlights: vec!["Built an embedding pipeline in Rust".into()],
            started_on: "2023-03-01".into(),
            salary: 90_000,
        }),
        past_positions: vec![
            Position {
                job_title: "Teaching assistant".into(),
                organization: Some("Universidad Nacional de La Plata".into()),
                highlights: vec![
                    "Ran physical chemistry labs".into(),
                    "Wrote the lab guides".into(),
                ],
                started_on: "2017-03-01".into(),
                salary: 20_000,
            },
            Position {
                job_title: "Freelance developer".into(),
                organization: None,
                highlights: vec![],
                started_on: "2015-06-01".into(),
                salary: 0,
            },
        ],
        skills: vec![
            Skill {
                name: "Rust".into(),
                level: Level::Expert,
            },
            Skill {
                name: "Python".into(),
                level: Level::Proficient,
            },
            Skill {
                name: "Qdrant".into(),
                level: Level::Familiar,
            },
        ],
        projects: BTreeMap::from([
            (
                "lvv".into(),
                Project {
                    name: "lvv".into(),
                    summary: "Embeds datasets with Ollama or OpenAI and loads them into Qdrant"
                        .into(),
                    technologies: vec!["rust".into(), "qdrant".into(), "ollama".into()],
                },
            ),
            (
                "spectra-search".into(),
                Project {
                    name: "spectra-search".into(),
                    summary: "Finds similar infrared spectra by cosine similarity".into(),
                    technologies: vec!["python".into(), "numpy".into()],
                },
            ),
        ]),
        interests: Some(vec![
            Interest("Information retrieval".into()),
            Interest("Spectroscopy".into()),
        ]),
        owner: "Ana Example".into(),
    }
}
