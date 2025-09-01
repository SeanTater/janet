# NOTE: Embedding-based runtime features have been removed from the MCP runtime.
# This document is retained for reference; embedding integration is available in the archived  crate.

# Repository Guidelines

## Project Structure & Module Organization

- Workspace crates live at the repository root: `janet-ai-context`, `janet-ai-gtk`, `janet-ai-mcp`, `janet-ai-retriever`.

Note: `janet-ai-embed` (embedding generation) has been removed from the MCP server's runtime
and is planned to be extracted from the core MCP flow. It remains a workspace crate but is no
longer required for MCP's `regex_search` and simplified `status` tools. Consider removing or
feature-gating `janet-ai-embed` if you want to eliminate embedding code entirely.
- Source: `*/src/`; tests: `*/tests/` or `*/src` (unit tests). Build artifacts: `target/`.

## Build, Test, and Development Commands

- `cargo build --workspace` — compile all crates.
- `cargo test --workspace` — run unit and integration tests across the workspace.
- `cargo run -p <crate>` — run a specific crate (e.g., `-p janet-ai-retriever`).
- `cargo fmt` / `cargo fmt --check` — enforce code formatting.
- `cargo clippy --all-targets --all-features -- -D warnings` — linting.
- `pre-commit install` / `pre-commit run --all-files` — run hooks configured in `.pre-commit-config.yaml`.
- `docker build -f Dockerfile .` — build container images (for CI or runtimes).

## Coding Style & Naming Conventions

- Follow Rust idioms: snake_case for filenames and function names; CamelCase for types; `SCREAMING_SNAKE_CASE` for constants.
- Indentation and formatting are enforced by `rustfmt` (`.rustfmt.toml` present).
- Use `clippy` and the project's `.clippy.toml` for lints; the repo treats warnings as errors in CI.

## Testing Guidelines

- Tests use Rust's built-in framework. Place unit tests inside `src` modules and integration tests in `tests/`.
- Run a single crate's tests: `cargo test -p janet-ai-context`.
- Test names should be descriptive (e.g., `fn parse_chunks_handles_empty_input()`).
- There is no strict coverage threshold in repo files; aim for thorough coverage for new features.

## Commit & Pull Request Guidelines

- Follow Conventional Commits. A `.gitmessage` template is included. Example: `feat(storage): add SQLite-based embedding store`.
- PR checklist: link related issue(s), include a concise description, screenshots if UI changes, ensure CI and pre-commit pass, and request at least one review.

## Security & Config Tips

- Never commit secrets; use environment variables. CI runs `cargo audit` and license checks.

If you need more detail, see `DEVELOPMENT.md` and the repository README.
