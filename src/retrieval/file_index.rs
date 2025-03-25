use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use super::analyzer::AnalyzerTrait;
use anyhow::Result;
use tokio::sync::mpsc;

#[derive(bincode::Encode, bincode::Decode)]
struct FileRef {
    /// The path to the file, relative to the root of the project
    relative_path: String,
    /// The file content, not interpreted as a string, to avoid fouling hashes
    content: Vec<u8>,
    /// The blake3 hash of the file
    hash: [u8; 32],
}

struct ChunkRecordV1 {
    chunk_id: u64,
    file_hash: [u8; 32],
    line_start: u32,
    line_end: u32,
}

pub struct FileIndex {
    keyspace: fjall::TransactionalKeyspace,
}

impl FileIndex {
    pub fn open(path: Option<&Path>) -> Result<Self> {
        let path = path.map(PathBuf::from).unwrap_or(".code-assistant".into());
        let keyspace = fjall::Config::new(path).open_transactional()?;
        Ok(Self { keyspace })
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

struct DirectoryTracker {
    base: PathBuf,
    events_tx: mpsc::Sender<PathBuf>,
    event_watcher: notify_debouncer_mini::Debouncer<notify::RecommendedWatcher>,
    listener: tokio::task::JoinHandle<Result<()>>,
}

impl DirectoryTracker {
    pub async fn open<T: AnalyzerTrait + 'static>(
        path: Option<&Path>,
        analyzer: T,
    ) -> Result<Self> {
        let base = path.map(PathBuf::from).unwrap_or(".".into());
        let (events_tx, events_rx) = mpsc::channel(128);
        let local_tx = events_tx.clone();

        // The listener needs to exist first or the rescan will probably block
        let listener = tokio::task::spawn(Self::listen(base.clone(), events_rx, analyzer));

        // Enqueue the initial scan before the events
        Self::rescan(&base, &events_tx).await?;

        // Introduce debouncing to reduce duplicate events
        // The longer scale is to avoid catching blips during git checkouts
        let mut event_watcher = notify_debouncer_mini::new_debouncer(
            Duration::from_secs(5),
            move |res: notify_debouncer_mini::DebounceEventResult| {
                // Not sure if eating all errors is the best approach but i'm not sure what the alternative here is,
                // except maybe to move to pollwatcher
                for ev in res.ok().into_iter().flatten() {
                    // This is not safe to call in an async context, but it is not run in async, it's in a different thread.
                    local_tx
                        .blocking_send(ev.path)
                        .expect("Analysis pipeline died while receiving file changes")
                }
            },
        )?;

        event_watcher
            .watcher()
            .watch(&base, notify::RecursiveMode::Recursive)?;

        Ok(Self {
            base,
            events_tx,
            event_watcher,
            listener,
        })
    }

    pub async fn rescan(base: &Path, events_tx: &mpsc::Sender<PathBuf>) -> Result<()> {
        for entry in ignore::Walk::new(base) {
            let entry = entry?;
            events_tx.send(entry.into_path()).await?;
        }
        Ok(())
    }

    /// A generic listener method that accepts any analyzer implementing AnalyzerTrait.
    async fn listen<A: AnalyzerTrait + 'static>(
        base: PathBuf,
        mut events_rx: mpsc::Receiver<PathBuf>,
        analyzer: A,
    ) -> Result<()> {
        while let Some(event) = events_rx.recv().await {
            analyzer.analyze(&base.join(&event)).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::retrieval::analyzer::MockAnalyzer;

    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use tokio::time::{sleep, Duration};

    /// Test the listener using the MockAnalyzer.
    #[tokio::test]
    async fn test_listen_with_mock_analyzer() -> Result<()> {
        // Set a base directory for the test.
        let base = PathBuf::from("/tmp/test_base");

        // Create a channel for file events.
        let (tx, rx) = mpsc::channel(128);

        // Create a new mock analyzer.
        let mock_analyzer = MockAnalyzer::new();
        // Clone the Arc so we can inspect it later.
        let calls_clone = Arc::clone(&mock_analyzer.calls);

        // Spawn the listener task using the generic listener.
        let listener = tokio::spawn(DirectoryTracker::listen(base.clone(), rx, mock_analyzer));

        // Send a couple of fake file events.
        let event1 = PathBuf::from("file1.txt");
        let event2 = PathBuf::from("subdir/file2.txt");
        tx.send(event1.clone()).await?;
        tx.send(event2.clone()).await?;
        // Close the sender so that the listener loop terminates.
        drop(tx);

        // Wait a short while to let the listener process the events.
        sleep(Duration::from_millis(50)).await;
        listener.await??;

        // Now check that the mock analyzer recorded the calls with full paths.
        let calls = calls_clone.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert!(
            calls.contains(&base.join(&event1)),
            "Expected path {} not found in calls: {:?}",
            base.join(&event1).display(),
            *calls
        );
        assert!(
            calls.contains(&base.join(&event2)),
            "Expected path {} not found in calls: {:?}",
            base.join(&event2).display(),
            *calls
        );

        Ok(())
    }
}
