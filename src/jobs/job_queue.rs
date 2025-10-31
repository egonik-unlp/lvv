use crate::{cache::cache_embeddings::Cache, jobs::job::Job};

#[derive(Debug, Clone, Default)]
pub struct JobQueue {
    queue: Vec<Job>,
    cache: Option<Cache>,
}
impl JobQueue {
    pub fn new() -> Self {
        JobQueue::default()
    }
    pub fn from_vec(vec: Vec<Job>) -> Self {
        JobQueue {
            queue: vec,
            ..Default::default()
        }
    }
    pub fn with_cache(&mut self, cache: Cache) -> &mut Self {
        self.cache = Some(cache);
        self
    }
    pub fn build(&mut self) {
        if let Some(cache) = self.clone().cache {
            self.queue.iter_mut().for_each(|job| {
                job.embedding = cache
                    .get_embedding(job.get_model(), job.clone().dataset.data.unwrap())
                    .map(|embedding| embedding.to_owned());
            })
        }
    }
    pub fn run(&mut self) -> anyhow::Result<()> {
        todo!()
    }
}
