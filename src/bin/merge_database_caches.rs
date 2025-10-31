use std::{collections::HashMap, fs::OpenOptions, io::Write};

use lvv::cache::cache_embeddings::Cache;
fn main() {
    let mut cache_1 = Cache::from_json_file("database_old.json").unwrap();
    let mut cache_2 = Cache::from_json_file("database.json").unwrap();
    cache_1.cache.extend(cache_2.cache);
    let st = serde_json::to_string(&cache_1).unwrap();
    let mut dump_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("database_joined.json")
        .unwrap();
    dump_file.write_all(st.as_bytes()).unwrap();
}
