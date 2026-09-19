use super::error::{CompletionError, RunError};

/// What happened to one record in [`Llm::run`](super::Llm::run).
#[derive(Debug)]
pub enum Outcome {
    /// The model's output was applied to the record.
    Applied,
    /// An output from the cache was applied; the model wasn't called.
    Cached,
    /// The record got no output and was left unchanged.
    Failed(CompletionError),
}

/// A snapshot of a run's progress, passed to
/// [`on_progress`](super::Run::on_progress) after each record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Progress {
    /// Records in the run.
    pub total: usize,
    /// Records with an outcome so far.
    pub done: usize,
    /// Records completed by the model.
    pub succeeded: usize,
    /// Records completed from the cache.
    pub cached: usize,
    /// Records that failed.
    pub failed: usize,
    /// Tokens used so far, as reported by the backend.
    pub tokens: u64,
}

/// How many records ended in each [`Outcome`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Completed by the model.
    pub applied: usize,
    /// Completed from the cache.
    pub cached: usize,
    /// Failed.
    pub failed: usize,
}

/// The outcome of every record in a run, in record order.
#[derive(Debug)]
pub struct Report {
    pub(crate) outcomes: Vec<Outcome>,
    pub(crate) tokens: u64,
}

impl Report {
    /// One outcome per record, in record order.
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    /// The failed records, as `(index, error)`.
    pub fn failures(&self) -> impl Iterator<Item = (usize, &CompletionError)> {
        self.outcomes
            .iter()
            .enumerate()
            .filter_map(|(index, outcome)| match outcome {
                Outcome::Failed(err) => Some((index, err)),
                _ => None,
            })
    }

    /// Whether every record got an output.
    pub fn is_complete(&self) -> bool {
        self.failures().next().is_none()
    }

    /// `Ok` when every record got an output.
    ///
    /// # Errors
    ///
    /// Returns [`RunError::Incomplete`] describing the first failure.
    pub fn ensure_all(&self) -> Result<(), RunError> {
        let failed = self.failures().count();
        match self.failures().next() {
            None => Ok(()),
            Some((first_index, err)) => Err(RunError::Incomplete {
                failed,
                total: self.outcomes.len(),
                first_index,
                first_error: err.to_string(),
            }),
        }
    }

    /// Tokens used, as reported by the backend.
    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    /// How many records ended in each outcome.
    pub fn counts(&self) -> Counts {
        let mut counts = Counts::default();
        for outcome in &self.outcomes {
            match outcome {
                Outcome::Applied => counts.applied += 1,
                Outcome::Cached => counts.cached += 1,
                Outcome::Failed(_) => counts.failed += 1,
            }
        }
        counts
    }

    /// The outcomes, taking ownership of the errors.
    pub fn into_outcomes(self) -> Vec<Outcome> {
        self.outcomes
    }
}
