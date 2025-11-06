use std::{fs::OpenOptions, io::Write};

use clap::Parser;
use lvv::cache::cache_embeddings::Cache;
fn main() {
    let eve = LvvExtendCache::parse();
    println!("{eve:?}");
    let mut file_base =
        Cache::from_json_file(eve.base.as_str()).expect("No se encontro archivo base");
    eve.others
        .into_iter()
        .map(|file| Cache::from_json_file(&file).unwrap())
        .for_each(|cache| file_base.cache.extend(cache.cache));
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open("database_joined_2.json")
        .unwrap();

    let st = serde_json::to_string(&file_base).unwrap();
    file.write_all(st.as_bytes()).unwrap();
}

#[derive(Debug, Parser)]
pub struct LvvExtendCache {
    pub base: String,
    pub others: Vec<String>,
}
