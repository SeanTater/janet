# Development Guide

This document outlines the development workflow and best practices for the Janet AI project.

## Quick Start

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up pre-commit hooks
uv tool install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg

# Verify everything works
cargo test
cargo clippy
cargo fmt --check
```

## Development Workflow

### Essential Commands

```bash
# Format code
cargo fmt

# Check code quality
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
cargo test

# Build project
cargo build

# Run individual crate tests
cargo test -p janet-ai-context
cargo test -p janet-ai-retriever

# Build release binaries
cargo build --release
```

### Pre-commit Hooks

Pre-commit hooks automatically run on every commit to ensure code quality:

- **Format checking**: `cargo fmt --check`
- **Linting**: `cargo clippy` with warnings as errors
- **File cleanup**: Remove trailing whitespace, ensure final newlines
- **Config validation**: Check YAML and TOML syntax
- **Conventional commits**: Enforce commit message format

If pre-commit fails, fix the issues and commit again. Most formatting issues can be auto-fixed:

```bash
# Auto-fix formatting
cargo fmt

# Auto-fix some clippy issues
cargo clippy --fix --allow-dirty --allow-staged
```

### Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert`

**Scopes**: `janet-ai-context`, `janet-ai-retriever`, `storage`, `ci`, `docs`

Example:
```
feat(storage): add SQLite-based embedding store

Implement EmbeddingStore trait with in-memory cosine similarity
search for vectors stored as BLOBs in SQLite database.

Closes #123
```

## CI/CD Pipeline

### GitHub Actions Workflows

The CI pipeline runs on every push and PR:

- **Multi-platform testing**: Linux, macOS, Windows
- **Multiple Rust versions**: stable, beta, MSRV (1.70.0)
- **Code quality checks**: format, clippy, tests
- **Security scanning**: cargo audit, dependency licenses
- **Documentation**: API docs deployed to GitHub Pages

### Release Process

1. Update version numbers in `Cargo.toml` files
2. Update `CHANGELOG.md` with new features and fixes
3. Create release PR with version bump
4. Merge to main after approval
5. Create git tag in format `v1.2.3`
6. GitHub Actions handles the rest automatically:
   - Builds cross-platform binaries
   - Creates GitHub release
   - Publishes to crates.io

## Project Structure

```
janet-ai/
├── janet-ai-context/           # Text chunking library
│   ├── src/
│   │   ├── bin/janet_context_cli.rs  # CLI tool
│   │   └── text.rs             # Core chunking logic
│   └── Cargo.toml
├── janet-ai-retriever/         # Storage and retrieval system
│   ├── src/
│   │   ├── storage/            # SQLite-based storage
│   │   └── retrieval/          # File indexing and analysis
│   └── Cargo.toml
└── .github/workflows/          # CI/CD pipelines
```

## Security Practices

- **No secrets in code**: Use environment variables
- **License compliance**: Only MIT, Apache-2.0, BSD licenses allowed
- **Dependency audits**: Automated vulnerability scanning
- **Supply chain validation**: Source and license verification

## Troubleshooting

### Common Issues

- **Format check fails**: Run `cargo fmt`
- **Clippy warnings**: Run `cargo clippy --fix --allow-dirty --allow-staged`
- **Test failures**: Run `cargo test -- --nocapture` for detailed output
- **Pre-commit issues**: Run `pre-commit run --all-files` to check all files

### Build Issues

- **Clean build**: `cargo clean && cargo build`
- **Update dependencies**: `cargo update`
- **Check individual crates**: `cargo check -p janet-ai-context`

### Pre-commit Setup Issues

```bash
# Reinstall pre-commit hooks
pre-commit uninstall
pre-commit install
pre-commit install --hook-type commit-msg

# Test hooks manually
pre-commit run --all-files
```

## Contributing

### Merge Requirements

- [ ] All CI checks pass
- [ ] Pre-commit hooks pass
- [ ] At least one approving review
- [ ] Conventional commit format
- [ ] No breaking changes without documentation

### Local Development Best Practices

- Use feature branches for all changes
- Keep commits focused and atomic
- Write tests for new functionality
- Update documentation for public APIs
- Run the full test suite before pushing

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [pre-commit documentation](https://pre-commit.com/)
- [uv documentation](https://docs.astral.sh/uv/)