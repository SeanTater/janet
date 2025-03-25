use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use super::analyzer::AnalyzerTrait;
use anyhow::Result;
use tokio::sync::mpsc;

use futures::stream::{self, StreamExt};

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
        let listener = tokio::task::spawn(Self::listen(events_rx, analyzer));

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
        events_rx: mpsc::Receiver<PathBuf>,
        analyzer: A,
    ) -> Result<()> {
        let analyzer_ref = &analyzer;
        let recv_stream = tokio_stream::wrappers::ReceiverStream::new(events_rx);
        recv_stream
            .for_each_concurrent(16, |event| async move {
                analyzer_ref.analyze(&event).await.unwrap_or_else(|err| {
                    tracing::error!("Failed to analyze path {}: {}", event.display(), err)
                })
            })
            .await;
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
    use tracing_test::traced_test;

    /// Test the listener using the MockAnalyzer.
    #[traced_test]
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
        let listener = tokio::spawn(DirectoryTracker::listen(rx, mock_analyzer));

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
            calls.contains(&event1),
            "Expected path {} not found in calls: {:?}",
            event1.display(),
            *calls
        );
        assert!(
            calls.contains(&event2),
            "Expected path {} not found in calls: {:?}",
            event2.display(),
            *calls
        );

        Ok(())
    }
}
