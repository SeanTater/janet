# Janet Context
Builds contexts for retrieval models (like BERT) as part of a RAG retrieval system.

The overall goal is to make code snippets that look like this:

```rs
passage: {"repo": "example", "path": "the_crate/src/lib.rs"}

context: {"module": "self"}
/// Create a wrapped database connection
use std::sync::{Arc, Mutex};
use anyhow::Result;
use some_crate::module::DBConnection;
use other_crate::stuff;
use crate::sync::DatabaseHandle;
...

context: {"module": "crate::sync"}
struct DatabaseHandle {
    thing: Arc<Mutex<DBConnection>>,
    ...
}

focus: {}
pub fn connect(host: str) -> Result<DatabaseHandle> {
    DatabaseHandle {
        thing: Arc::new(Mutex::new(DBConnection::connect(host)?))
    }
}

```

Highlights:
* Start with the model's "document" prefix (often either "document:" or "passage:")
* Continue with file level background info (useful for searches mentioning filename)
* Give segments of context relevant to this code under focus
* Finally, give the code under focus


## The ContextBuilder Trait

