//! Realistic end-to-end use of `#[derive(VectorDatabase)]` + `#[derive(VectorDatabaseItem)]`.
//!
//! Scenario: a personal "portfolio" knowledge base (CV + projects + papers) that
//! gets chunked into one vector point per item, embedded by description, and
//! uploaded to Qdrant with the item's fields as payload — the kind of thing you
//! would put behind a "ask my CV" RAG chatbot.
//!
//! Everything relies on the derives. The only hand-written part is what a
//! library user is expected to provide: `IntoDescriptionValue` for their own
//! types used as `#[lvv(description)]` fields (`Level`, `Organization`).
//!
//! ```text
//! cargo run -p lvv --bin portfolio_rag                         # dry run: print drafts
//! cargo run -p lvv --bin portfolio_rag -- --input me.json      # load portfolio from JSON
//! cargo run -p lvv --bin portfolio_rag -- --embed              # + embed descriptions via Ollama
//! cargo run -p lvv --bin portfolio_rag -- --embed --qdrant http://localhost:6334
//! ```

use std::collections::BTreeMap;

use chrono::NaiveDate;
use clap::Parser;
use lvv::{
    db::{
        Distance, QdrantSink, Sink, SinkContext,
        vector_database::{DatabaseParams, Location},
    },
    inference::EmbeddingProvider,
    points::{IntoDescriptionValue, VectorDatabase, VectorDatabaseItem, VectorPointDraft},
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Items: each one becomes (at least) one vector point.
// ---------------------------------------------------------------------------

/// Who the portfolio belongs to. Exactly one per portfolio -> scalar container field.
#[derive(Debug, Clone, Serialize, Deserialize, VectorDatabaseItem)]
struct Contact {
    #[lvv(description)]
    name: String,
    #[lvv(description)]
    headline: String,
    email: String,
    /// Private: must never reach the vector store.
    #[lvv(skip)]
    phone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Level {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

impl IntoDescriptionValue for Level {
    fn into_description_value(&self) -> String {
        match self {
            Level::Beginner => "beginner",
            Level::Intermediate => "intermediate",
            Level::Advanced => "advanced",
            Level::Expert => "expert",
        }
        .to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, VectorDatabaseItem)]
struct Skill {
    #[lvv(description)]
    name: String,
    /// A non-String description field (user-defined enum).
    #[lvv(description)]
    level: Level,
    /// Payload key should be `years`.
    #[lvv(rename = "years")]
    years_of_experience: Option<u8>,
    /// Upstream DB id, useless for retrieval.
    #[lvv(skip)]
    internal_id: u64,
}

/// Plain nested struct (not an item itself), used inside several items.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Organization {
    name: String,
    url: Option<String>,
}

impl IntoDescriptionValue for Organization {
    /// Only the name is worth embedding; the URL stays in the payload.
    fn into_description_value(&self) -> String {
        self.name.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, VectorDatabaseItem)]
struct Position {
    /// Two commands in a single attribute: part of the description AND renamed.
    #[lvv(description, rename = "role")]
    title: String,
    /// Nested struct as a description field.
    #[lvv(description)]
    organization: Organization,
    started: NaiveDate,
    ended: Option<NaiveDate>,
    /// Collection as a description field.
    #[lvv(description)]
    highlights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, VectorDatabaseItem)]
struct Project {
    #[lvv(description)]
    name: String,
    #[lvv(description)]
    summary: String,
    #[lvv(rename = "stack")]
    technologies: Vec<String>,
    /// serde attributes the payload should keep honouring.
    #[serde(skip_serializing_if = "Option::is_none")]
    repository: Option<String>,
    /// Two separate attributes on one field.
    #[lvv(skip)]
    #[serde(default)]
    private_notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, VectorDatabaseItem)]
struct Publication {
    #[lvv(description)]
    title: String,
    /// Serialized as `abstract` by serde — the payload should agree.
    #[serde(rename = "abstract")]
    #[lvv(description)]
    abstract_text: Option<String>,
    authors: Vec<String>,
    year: u16,
    doi: Option<String>,
}

/// Generic item: the issuer can be an organization or just a name.
#[derive(Debug, Clone, Serialize, Deserialize, VectorDatabaseItem)]
struct Credential<I> {
    #[lvv(description)]
    name: String,
    #[lvv(description)]
    issuer: I,
    issued: NaiveDate,
}

/// Newtype (tuple struct) item.
#[derive(Debug, Clone, Serialize, Deserialize, VectorDatabaseItem)]
struct Interest(#[lvv(description)] String);

// ---------------------------------------------------------------------------
// Container: one struct describing the whole database.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, VectorDatabase)]
struct Portfolio {
    /// Scalar item.
    owner: Contact,
    /// Plain collection.
    skills: Vec<Skill>,
    /// Optional scalar item (you may be between jobs).
    current_position: Option<Position>,
    /// Fully qualified collection path.
    past_positions: std::vec::Vec<Position>,
    projects: Vec<Project>,
    /// Optional collection.
    publications: Option<Vec<Publication>>,
    /// Collection of a generic item.
    credentials: Vec<Credential<Organization>>,
    interests: Vec<Interest>,
    /// Bookkeeping, not an item. Same `skip` spelling as on items.
    #[lvv(skip)]
    updated_at: NaiveDate,
}

// ---------------------------------------------------------------------------
// CLI + pipeline
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "Turn a portfolio into vector points using the lvv derives")]
struct Args {
    /// Load the portfolio from a JSON file instead of the built-in sample.
    #[arg(long)]
    input: Option<String>,
    /// Embed each point's description with Ollama.
    #[arg(long)]
    embed: bool,
    /// Ollama embedding model.
    #[arg(long, default_value = "embeddinggemma")]
    model: String,
    /// Upload to this Qdrant gRPC URL (requires --embed).
    #[arg(long)]
    qdrant: Option<String>,
    /// Prefix for the per-category Qdrant collections.
    #[arg(long, default_value = "portfolio")]
    collection_prefix: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let portfolio = match &args.input {
        Some(path) => serde_json::from_str::<Portfolio>(&std::fs::read_to_string(path)?)?,
        None => sample_portfolio(),
    };

    // 1. Container -> drafts.
    let drafts = portfolio.point_drafts()?;
    println!("{} point drafts generated", drafts.len());

    // 2. Items can also be used on their own (e.g. re-index a single edited project).
    let single = portfolio.projects[0].try_into_database_item()?;
    println!("single project re-index -> category {:?}", single.category);

    // 3. One Qdrant collection per category.
    let mut by_category: BTreeMap<String, Vec<VectorPointDraft>> = BTreeMap::new();
    for draft in drafts {
        by_category
            .entry(draft.category.clone())
            .or_default()
            .push(draft);
    }

    for (category, drafts) in &by_category {
        println!("\n== {category} ({} points) ==", drafts.len());
        for draft in drafts {
            println!("  description: {:?}", draft.description);
            println!("  payload:     {}", serde_json::to_string(&draft.payload)?);
        }
    }

    // 4. Pre-flight: what we expect the derives to have produced before we
    //    push anything to a shared vector store.
    preflight(&by_category)?;

    if !args.embed {
        println!("\n(dry run — pass --embed to call Ollama)");
        return Ok(());
    }

    // 5. Embed descriptions, store payloads.
    let embedder = EmbeddingProvider::new(&args.model);
    for (category, drafts) in by_category {
        let descriptions: Vec<&str> = drafts.iter().map(|d| d.description.as_str()).collect();
        let embeddings = embedder.embed_texts(&descriptions).await?;
        let dims = embeddings.first().map_or(0, Vec::len) as u64;
        println!(
            "\nembedded {} `{category}` descriptions ({dims} dims)",
            embeddings.len()
        );

        let Some(url) = &args.qdrant else { continue };
        let rows = drafts
            .iter()
            .map(|d| serde_json::to_value(&d.payload))
            .collect::<Result<Vec<_>, _>>()?;
        let collection = format!("{}_{}", args.collection_prefix, category.to_lowercase());
        // `Location::new_local` wants a `&'static str`.
        let url: &'static str = Box::leak(url.clone().into_boxed_str());
        let sink = QdrantSink::new(DatabaseParams::new(
            Location::new_local(url),
            collection.clone(),
            Distance::Cosine,
            dims as u16,
        ));
        sink.write(&SinkContext {
            collection_name: &collection,
            dims,
            extends: true,
            embeddings: &embeddings,
            rows: &rows,
        })
        .await?;
        println!("uploaded to qdrant collection `{collection}`");
    }
    Ok(())
}

/// Checks the invariants the attributes promise. Reports every problem instead
/// of stopping at the first one.
fn preflight(
    by_category: &BTreeMap<String, Vec<VectorPointDraft>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut problems = Vec::new();
    let payload_of = |category: &str| -> Option<serde_json::Value> {
        by_category
            .get(category)
            .and_then(|drafts| drafts.first())
            .map(|d| serde_json::to_value(&d.payload).unwrap_or_default())
    };
    let mut expect = |ok: bool, what: String| {
        if !ok {
            problems.push(what);
        }
    };

    for category in [
        "Contact",
        "Skill",
        "Position",
        "Project",
        "Publication",
        "Credential",
        "Interest",
    ] {
        expect(
            by_category.contains_key(category),
            format!(
                "no points with category {category:?} (got {:?})",
                by_category.keys().collect::<Vec<_>>()
            ),
        );
    }
    for (category, drafts) in by_category {
        for d in drafts {
            expect(
                !d.description.trim().is_empty(),
                format!("{category}: empty description"),
            );
            expect(
                serde_json::to_value(&d.payload).is_ok_and(|p| p.get("category").is_some()),
                format!("{category}: payload has no `category` key"),
            );
        }
    }
    let has =
        |category: &str, key: &str| payload_of(category).is_some_and(|p| p.get(key).is_some());
    expect(
        !has("Contact", "phone"),
        "Contact: skipped `phone` leaked into payload".into(),
    );
    expect(
        !has("Skill", "internal_id"),
        "Skill: skipped `internal_id` leaked into payload".into(),
    );
    expect(
        has("Skill", "years"),
        "Skill: rename -> `years` not applied".into(),
    );
    expect(
        !has("Skill", "years_of_experience"),
        "Skill: original name `years_of_experience` still present".into(),
    );
    expect(
        has("Position", "role"),
        "Position: rename -> `role` not applied (stacked attribute)".into(),
    );
    expect(
        has("Project", "stack"),
        "Project: rename -> `stack` not applied".into(),
    );
    expect(
        !has("Project", "private_notes"),
        "Project: skipped `private_notes` leaked into payload".into(),
    );
    expect(
        has("Publication", "abstract"),
        "Publication: #[serde(rename = \"abstract\")] ignored in payload".into(),
    );
    expect(
        has("Interest", "0")
            || payload_of("Interest").is_some_and(|p| !p.as_object().is_some_and(|o| o.len() <= 1)),
        "Interest: newtype field missing from payload".into(),
    );

    let skill_desc = by_category
        .get("Skill")
        .and_then(|d| d.first())
        .map(|d| d.description.as_str());
    expect(
        skill_desc.is_some_and(|d| d.contains("Rust") && d.to_lowercase().contains("expert")),
        format!("Skill: description should mention name and level, got {skill_desc:?}"),
    );

    if problems.is_empty() {
        println!("\npreflight: all checks passed");
    } else {
        println!("\npreflight: {} problem(s)", problems.len());
        for p in &problems {
            println!("  - {p}");
        }
    }
    Ok(())
}

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
}

fn sample_portfolio() -> Portfolio {
    let unlp = Organization {
        name: "Universidad Nacional de La Plata".into(),
        url: Some("https://unlp.edu.ar".into()),
    };
    Portfolio {
        owner: Contact {
            name: "Ana Example".into(),
            headline: "Research software engineer — embeddings & spectroscopy".into(),
            email: "ana@example.org".into(),
            phone: "+54 221 555 0100".into(),
        },
        skills: vec![
            Skill {
                name: "Rust".into(),
                level: Level::Expert,
                years_of_experience: Some(6),
                internal_id: 101,
            },
            Skill {
                name: "Qdrant".into(),
                level: Level::Intermediate,
                years_of_experience: None,
                internal_id: 103,
            },
        ],
        current_position: Some(Position {
            title: "Senior Engineer".into(),
            organization: Organization {
                name: "Lensing Lab".into(),
                url: None,
            },
            started: date(2023, 3, 1),
            ended: None,
            highlights: vec!["Built the lvv embedding pipeline".into()],
        }),
        past_positions: vec![Position {
            title: "Teaching Assistant".into(),
            organization: unlp.clone(),
            started: date(2017, 3, 1),
            ended: Some(date(2022, 12, 31)),
            highlights: vec!["Physical chemistry labs".into()],
        }],
        projects: vec![Project {
            name: "lvv".into(),
            summary: "Embed datasets with Ollama/OpenAI and load them into Qdrant".into(),
            technologies: vec!["rust".into(), "qdrant".into(), "ollama".into()],
            repository: Some("https://github.com/egonik-unlp/lvv".into()),
            private_notes: "derive macros still WIP".into(),
        }],
        publications: Some(vec![Publication {
            title: "Vector retrieval for vibrational spectra".into(),
            abstract_text: Some(
                "We embed IR spectra and retrieve analogues by cosine similarity.".into(),
            ),
            authors: vec!["A. Example".into(), "B. Coauthor".into()],
            year: 2025,
            doi: None,
        }]),
        credentials: vec![Credential {
            name: "PhD in Chemistry".into(),
            issuer: unlp,
            issued: date(2022, 11, 15),
        }],
        interests: vec![Interest("information retrieval".into())],
        updated_at: date(2026, 9, 13),
    }
}
