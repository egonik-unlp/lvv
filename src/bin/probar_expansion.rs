use lvv::points::{VectorDatabase, VectorDatabaseItem};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Skill {
    name: String,
    level: String,
}

impl VectorDatabaseItem for Skill {
    fn category(&self) -> String {
        "skill".into()
    }

    fn into_description(&self) -> String {
        format!("{} - {}", self.name, self.level)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Project {
    name: String,
    description: String,
}

impl VectorDatabaseItem for Project {
    fn category(&self) -> String {
        "project".into()
    }

    fn into_description(&self) -> String {
        format!("{}: {}", self.name, self.description)
    }
}

#[derive(VectorDatabase)]
struct TestDatabase {
    skills: Vec<Skill>,
    projects: Vec<Project>,
}

fn main() {
    let db = TestDatabase {
        skills: vec![
            Skill {
                name: "Rust".into(),
                level: "advanced".into(),
            },
            Skill {
                name: "Python".into(),
                level: "advanced".into(),
            },
        ],

        projects: vec![Project {
            name: "Test project".into(),
            description: "Testing VectorDatabase expansion".into(),
        }],
    };

    // Adjust this to whatever method VectorDatabase exposes.
    let points = db.point_drafts().expect("failed to generate vector points");

    for point in points {
        println!("{point:#?}");
    }
}
