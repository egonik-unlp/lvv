use async_openai::{
    Client,
    types::{CreateCompletionRequestArgs, CreateEmbeddingRequestArgs},
};

pub async fn create_openai_embeddings(model: &str) -> anyhow::Result<()> {
    let client = Client::new();
    // let request = CreateEmbeddingRequestArgs::default().model(valua)

    todo!()
}
