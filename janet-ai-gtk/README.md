# Janet AI GTK

A modern GTK4 desktop application for semantic code search and AI-powered repository exploration. Janet AI GTK provides an intuitive chat interface for searching your codebase using both regex patterns and semantic similarity.

## For Users

### What is Janet AI GTK?

Janet AI GTK is a desktop application that lets you search and understand your code using natural language. Instead of remembering exact function names or complex grep patterns, you can ask questions like "authentication logic" or "error handling" and get relevant code results.

### Features

- **🎯 Semantic Search**: Find code by meaning, not just exact text matches
- **🔍 Regex Search**: Traditional pattern-based search for precise queries
- **💬 Chat Interface**: Intuitive conversational interface for exploring code
- **📁 Repository Selection**: Easy folder selection or command-line repository specification
- **⚡ Real-time Results**: Fast search with immediate response display
- **🎨 Modern UI**: Clean, modern GTK4 interface with syntax highlighting

### Installation

#### Prerequisites

- GTK4 development libraries
- Rust toolchain (1.70+)

#### Ubuntu/Debian
```bash
sudo apt install libgtk-4-dev build-essential
```

#### Fedora
```bash
sudo dnf install gtk4-devel gcc
```

#### macOS
```bash
brew install gtk4
```

#### Build from Source
```bash
# Clone the repository
git clone https://github.com/SeanTater/janet.git
cd janet

# Build the GTK application
cargo build --release -p janet-ai-gtk

# The binary will be at target/release/janet-ai-gtk
```

### Getting Started

#### 1. Index Your Repository

Before using Janet AI GTK, you need to index your codebase:

```bash
# Index the current directory
cargo run -p janet-ai-retriever -- index --repo .

# Or index a specific repository
cargo run -p janet-ai-retriever -- index --repo /path/to/your/project
```

This creates a `.janet-ai.db` file containing the searchable index of your code.

#### 2. Launch the Application

**Option A: Select folder at startup**
```bash
./target/release/janet-ai-gtk
```
The app will show a folder selection dialog where you can choose your repository.

**Option B: Specify repository via command line**
```bash
./target/release/janet-ai-gtk /path/to/your/repo
```

#### 3. Start Searching

Once the app is running, you can search your code using:

- **Semantic search**: Just type your query
  - `authentication logic`
  - `database connection handling`
  - `error logging functions`

- **Regex search**: Prefix with `/regex`
  - `/regex fn.*authenticate`
  - `/regex class.*Controller`
  - `/regex TODO.*fix`

### Search Tips

- **Be descriptive**: Instead of `login`, try `user authentication flow`
- **Use context**: `HTTP error handling in API routes` vs just `error`
- **Combine approaches**: Use semantic search to find relevant files, then regex for specific patterns
- **Experiment**: The AI understands synonyms and related concepts

### Troubleshooting

**"No results found"**
- Ensure your repository is indexed: `cargo run -p janet-ai-retriever -- index --repo .`
- Try broader search terms
- Check that `.janet-ai.db` exists in your repository root

**Application won't start**
- Verify GTK4 is installed
- Check that the repository path is valid
- Ensure you have read permissions for the repository

**Search is slow**
- Large repositories may take time to search
- Consider indexing only specific subdirectories
- Check available disk space (indexing creates temporary files)

## For Developers

### Architecture

Janet AI GTK is built using:

- **GTK4**: Modern Linux desktop UI framework
- **Relm4**: Reactive GTK4 framework for Rust
- **janet-ai-mcp**: MCP (Model Context Protocol) server for search functionality
- **Tokio**: Async runtime for non-blocking operations

### Project Structure

```
janet-ai-gtk/
├── src/
│   ├── main.rs          # Application entry point and CLI parsing
│   ├── ui.rs            # Main UI components and message handling
│   └── style.css        # GTK CSS styling
├── Cargo.toml           # Dependencies and metadata
└── README.md           # This file
```

### Key Components

#### Application State (`App`)
- `server_config`: MCP server configuration for the current repository
- `current_repo`: Currently selected repository path
- `messages`: Chat message history
- UI widget references for dynamic updates

#### Message System (`AppMsg`)
- `SendMessage`: User initiated search
- `ShowResult`: Display search results
- `SelectFolder`: Open folder selection dialog
- `FolderSelected`: Process selected repository
- `ClearEntry`: Reset input field

#### UI Flow
1. **Startup**: Check for command-line repo argument
2. **Folder Selection**: Show selection dialog if no repo specified
3. **Chat Interface**: Switch to main chat UI after repo selection
4. **Search Processing**: Handle user queries asynchronously
5. **Result Display**: Show formatted search results

### Development Setup

```bash
# Clone repository
git clone https://github.com/SeanTater/janet.git
cd janet/janet-ai-gtk

# Install dependencies (see user installation section)

# Run in development mode
cargo run

# Run with a specific repository
cargo run -- /path/to/test/repo

# Run tests
cargo test

# Check code style
cargo clippy
cargo fmt
```

### Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Build with specific features
cargo build --features "some-feature"
```

### Code Style

- Follow Rust 2024 edition conventions
- Use `cargo fmt` for consistent formatting
- Run `cargo clippy` to catch common issues
- Prefer explicit error handling over panics
- Use inline string formatting: `println!("value: {value}")`

### Contributing

1. **Feature Requests**: Open an issue describing the feature and use case
2. **Bug Reports**: Include reproduction steps and environment details
3. **Pull Requests**:
   - Fork the repository
   - Create a feature branch
   - Implement changes with tests
   - Submit PR with description

### Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_folder_selection

# Run with output
cargo test -- --nocapture

# Integration tests (requires GTK display)
cargo test --test integration
```

### Debugging

- Use `RUST_LOG=debug cargo run` for detailed logging
- GTK Inspector: `GTK_DEBUG=interactive cargo run`
- Memory debugging: `valgrind --tool=memcheck ./target/debug/janet-ai-gtk`

### Performance Considerations

- **Async Operations**: All search operations are non-blocking
- **Memory Usage**: Chat history is kept in memory (consider limits for long sessions)
- **Index Size**: Large repositories create large indexes (monitor disk usage)
- **UI Responsiveness**: Long operations show loading states

### Dependencies

Key dependencies and their purposes:

- `gtk4` (0.9): GTK4 bindings for Rust
- `relm4` (0.9): Reactive GTK4 framework
- `janet-ai-mcp`: Local MCP server for search operations
- `clap` (4.5): Command-line argument parsing
- `tokio` (1.44): Async runtime
- `chrono` (0.4): Timestamp handling for messages
- `tracing` (0.1): Structured logging

### Release Process

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.1.0`
4. Build release binary: `cargo build --release`
5. Test release binary with various repositories
6. Push tag: `git push origin v0.1.0`

### IDE Integration

**VS Code**
- Install rust-analyzer extension
- Use "Rust: Run" command for quick testing

**CLion/IntelliJ**
- Install Rust plugin
- Configure run configurations for different repo paths

### Future Improvements

- **Themes**: Support for dark/light theme switching
- **History**: Persistent search history across sessions
- **Bookmarks**: Save frequently accessed search results
- **Multi-repo**: Support for searching across multiple repositories
- **Export**: Export search results to various formats
- **Collaboration**: Share search sessions with team members
