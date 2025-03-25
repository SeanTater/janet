use std::path::{Path, PathBuf};

use anyhow::Result;

#[derive(bincode::Encode, bincode::Decode)]
struct FileRef {
    /// The path to the file, relative to the root of the project
    relative_path: String,
    /// The file content, not interpreted as a string, to avoid fouling hashes
    content: Vec<u8>,
    /// The blake3 hash of the file
    hash: [u8; 32],
}

#[derive(Clone)]
pub struct FileIndex {
    pub(crate) base: PathBuf,
    keyspace: fjall::TransactionalKeyspace,
}

impl FileIndex {
    pub fn open(base: &Path) -> Result<Self> {
        let assist_dir = base.join(".code-assistant/index");
        std::fs::create_dir_all(&assist_dir)?;
        let keyspace = fjall::Config::new(assist_dir).open_transactional()?;
        Ok(Self {
            base: base.to_path_buf(),
            keyspace,
        })
    }

    fn file_refs(&self) -> Result<fjall::TransactionalPartition> {
        Ok(self
            .keyspace
            .open_partition("FileRef", fjall::PartitionCreateOptions::default())?)
    }

    pub fn get<BX: bincode::Encode + bincode::Decode<()>>(
        &self,
        table: &str,
        key: &[u8],
    ) -> Result<Option<BX>> {
        let part = self
            .keyspace
            .open_partition(table, fjall::PartitionCreateOptions::default())?;
        Ok(part
            .get(key)?
            .map(|item| bincode::decode_from_slice(&item, bincode::config::standard()))
            .transpose()?
            .map(|x| x.0))
    }

    pub fn upsert<
        BX: bincode::Encode + bincode::Decode<()>,
        Merger: Fn(Option<BX>) -> Option<BX>,
    >(
        &self,
        table: &str,
        key: &[u8],
        merge: Merger,
    ) -> Result<Option<BX>> {
        let part = self
            .keyspace
            .open_partition(table, fjall::PartitionCreateOptions::default())?;
        let config = bincode::config::standard();
        let new = part.update_fetch(key, |prev| {
            let prev = prev
                .and_then(|slc| bincode::decode_from_slice(slc, config).ok())
                .map(|x| x.0);
            let next = merge(prev)
                .and_then(|item| bincode::encode_to_vec(item, config).ok())
                .map(fjall::Slice::from);
            next
        })?;
        Ok(new
            .map(|slc| bincode::decode_from_slice(&slc, config))
            .transpose()?
            .map(|x| x.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use tempfile::tempdir;

    // A simple test type that can be encoded/decoded using bincode.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, bincode::Encode, bincode::Decode)]
    struct TestValue(u32);

    /// Test that a new key can be inserted and then updated.
    #[test]
    fn test_upsert_insert_and_update() -> Result<()> {
        // Create a temporary directory for the key-value store.
        let temp_dir = tempdir()?;
        let index = FileIndex::open(temp_dir.path())?;
        let key = b"test-key";

        // Insert an initial value using upsert.
        let inserted: Option<TestValue> =
            index.upsert("TestTable", key, |_| Some(TestValue(42)))?;
        assert_eq!(inserted, Some(TestValue(42)));

        // Update the value by doubling it.
        let updated: Option<TestValue> = index.upsert("TestTable", key, |prev| {
            prev.map(|TestValue(val)| TestValue(val * 2))
        })?;
        assert_eq!(updated, Some(TestValue(84)));

        // Verify that a subsequent get returns the updated value.
        let fetched: Option<TestValue> = index.get("TestTable", key)?;
        assert_eq!(fetched, Some(TestValue(84)));
        Ok(())
    }

    /// Test that a key that does not exist returns None.
    #[test]
    fn test_get_non_existent_key() -> Result<()> {
        let temp_dir = tempdir()?;
        let index = FileIndex::open(temp_dir.path())?;
        let key = b"non-existent";

        let fetched: Option<TestValue> = index.get("TestTable", key)?;
        assert!(fetched.is_none());
        Ok(())
    }

    /// Test that upsert can be used to delete a value by returning None from the merger.
    #[test]
    fn test_upsert_deletion() -> Result<()> {
        let temp_dir = tempdir()?;
        let index = FileIndex::open(temp_dir.path())?;
        let key = b"key-to-delete";

        // Insert an initial value.
        let _ = index.upsert("TestTable", key, |_| Some(TestValue(100)))?;
        let fetched: Option<TestValue> = index.get("TestTable", key)?;
        assert_eq!(fetched, Some(TestValue(100)));

        // Now "delete" the key by returning None.
        let deleted: Option<TestValue> = index.upsert("TestTable", key, |_| None)?;
        assert!(deleted.is_none());

        // Verify that the key is no longer present.
        let fetched_after: Option<TestValue> = index.get("TestTable", key)?;
        assert!(fetched_after.is_none());
        Ok(())
    }

    /// Test that using a different table name (for example "FileRef") works correctly.
    #[test]
    fn test_file_refs_partition() -> Result<()> {
        let temp_dir = tempdir()?;
        let index = FileIndex::open(temp_dir.path())?;
        let key = b"file1";
        let value = TestValue(1);

        // Upsert into the "FileRef" partition.
        let _ = index.upsert("FileRef", key, |_| Some(value))?;
        let fetched: Option<TestValue> = index.get("FileRef", key)?;
        assert_eq!(fetched, Some(value));
        Ok(())
    }
}
