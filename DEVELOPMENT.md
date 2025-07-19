# Development Guide

This document outlines the development workflow, CI/CD setup, and best practices for the Janet project.

## Quick Start

1. **Install development tools**:
   ```bash
   cargo install cargo-make
   cargo make install-tools
   ```

2. **Set up git configuration**:
   ```bash
   cargo make git-setup
   ```

3. **Run all checks locally**:
   ```bash
   cargo make ci
   ```

## Development Workflow

### Daily Development

```bash
# Check code quality before committing
cargo make pre-commit

# Run all CI checks locally
cargo make ci

# Build documentation and open in browser
cargo make doc-open
```

### Code Quality

- **Formatting**: We use `rustfmt` with custom configuration (`.rustfmt.toml`)
- **Linting**: Enforced via `clippy` with strict settings (`.clippy.toml`)
- **Testing**: All code must have tests; aim for high coverage
- **Documentation**: Public APIs must be documented

### Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert`

**Scopes**: `janet-context`, `janet-retriever`, `storage`, `ci`, `docs`

Example:
```
feat(storage): add SQLite-based embedding store

Implement EmbeddingStore trait with in-memory cosine similarity
search for vectors stored as BLOBs in SQLite database.

Closes #123
```

## CI/CD Pipeline

### GitHub Actions Workflows

#### 1. **CI Workflow** (`.github/workflows/ci.yml`)
Runs on every push and PR to main:

- **Multi-platform testing**: Linux, macOS, Windows
- **Multiple Rust versions**: stable, beta, MSRV (1.70.0)
- **Comprehensive checks**:
  - `cargo build --all-features`
  - `cargo test --all-features`
  - `cargo clippy -- -D warnings`
  - `cargo fmt --check`
  - `cargo doc --no-deps`
- **Code coverage**: Using `cargo-llvm-cov` with Codecov integration
- **Security audit**: `cargo audit` for vulnerability scanning
- **Workspace validation**: Ensures all crates build independently

#### 2. **Release Workflow** (`.github/workflows/release.yml`)
Triggered on version tags (`v*.*.*`):

- **Multi-platform binaries**: Creates optimized release builds
- **GitHub Releases**: Automatic release notes and asset uploads
- **Crates.io publishing**: Publishes both `janet-context` and `janet-retriever`
- **Asset packaging**: `.tar.gz` for Unix, `.zip` for Windows

#### 3. **Security Workflow** (`.github/workflows/security.yml`)
Runs weekly and on every push:

- **Vulnerability scanning**: `cargo audit` and `cargo deny`
- **License compliance**: Checks for compatible licenses only
- **Supply chain security**: Validates dependency sources
- **Secret scanning**: TruffleHog integration

#### 4. **Documentation Workflow** (`.github/workflows/docs.yml`)
Maintains project documentation:

- **API docs**: Built with `cargo doc` and deployed to GitHub Pages
- **Link checking**: Validates all markdown links
- **Spell checking**: Using `typos` for technical documents

#### 5. **Conventional Commits** (`.github/workflows/conventional-commits.yml`)
Enforces commit message standards:

- **Commit format validation**: Ensures conventional commit format
- **PR title checking**: Validates pull request titles
- **Scope validation**: Enforces allowed scopes

### Dependabot Configuration

Automated dependency updates (`.github/dependabot.yml`):

- **Weekly updates**: Monday 09:00 UTC
- **Grouped updates**: Related dependencies updated together
- **Cargo and GitHub Actions**: Both ecosystems covered
- **Review assignment**: Automatically assigns maintainers

## Security Practices

### Dependency Management

- **License compliance**: Only MIT, Apache-2.0, BSD licenses allowed
- **Vulnerability scanning**: Automated via `cargo audit`
- **Supply chain validation**: `cargo deny` checks sources and versions
- **Regular updates**: Dependabot keeps dependencies current

### Secret Management

- **No secrets in code**: Use environment variables
- **Secret scanning**: TruffleHog prevents accidental commits
- **Minimal permissions**: CI uses least-privilege access

### Security Audits

- **Weekly scans**: Automated vulnerability checks
- **RUSTSEC advisories**: Real-time security advisory monitoring
- **Dependency validation**: Source and license verification

## Local Development Setup

### Required Tools

Install these tools for full local development:

```bash
# Essential tools
cargo install cargo-make
cargo install cargo-audit
cargo install cargo-deny
cargo install cargo-llvm-cov

# Optional but recommended
cargo install cargo-outdated
cargo install cargo-license
```

### Pre-commit Hooks

Set up git hooks for automatic quality checks:

```bash
# Install pre-commit (Python tool)
pip install pre-commit

# Set up hooks (if we add .pre-commit-config.yaml)
pre-commit install
```

### Local CI Simulation

Run the same checks as CI:

```bash
# Full CI simulation
cargo make ci

# Individual checks
cargo make format-check
cargo make clippy
cargo make test
cargo make audit
cargo make deny
```

## Release Process

### Version Management

1. **Update version numbers** in `Cargo.toml` files
2. **Update CHANGELOG.md** with new features and fixes
3. **Create release PR** with version bump
4. **Merge to main** after approval
5. **Create git tag** in format `v1.2.3`
6. **GitHub Actions** handles the rest automatically

### Release Automation

The release workflow automatically:

- Builds cross-platform binaries
- Creates GitHub release with changelog
- Uploads binary assets
- Publishes to crates.io
- Generates release notes

### Hotfix Process

For critical fixes:

1. **Create hotfix branch** from latest release tag
2. **Apply minimal fix** with tests
3. **Version bump** patch number
4. **Fast-track review** and merge
5. **Tag immediately** to trigger release

## Performance Monitoring

### Benchmarks

- **Micro-benchmarks**: Use `criterion` for performance-critical code
- **Integration benchmarks**: Full workflow timing
- **Memory profiling**: Track memory usage patterns
- **Performance regression**: CI catches slowdowns

### Profiling

```bash
# CPU profiling
cargo install cargo-profiler
cargo profiler callgrind --bin janet-retriever

# Memory profiling
cargo install cargo-valgrind
cargo valgrind --tool=massif --bin janet-retriever
```

## Troubleshooting

### Common CI Failures

- **Format check fails**: Run `cargo make format` locally
- **Clippy warnings**: Run `cargo make clippy-fix` for auto-fixes
- **Test failures**: Run `cargo make test` with `--nocapture` for details
- **Coverage too low**: Add tests to increase coverage
- **Security audit fails**: Update dependencies or add exceptions

### Local Development Issues

- **SQLite errors**: Ensure file permissions and disk space
- **Build failures**: Clean with `cargo make clean` and rebuild
- **Test database conflicts**: Use `cargo test -- --test-threads=1`

### Dependency Issues

- **Version conflicts**: Use `cargo tree` to debug
- **License violations**: Check `cargo license` output
- **Audit failures**: Review `cargo audit` recommendations

## Contributing

### Code Review Checklist

- [ ] Follows conventional commit format
- [ ] All tests pass locally
- [ ] Documentation updated for public APIs
- [ ] No new clippy warnings
- [ ] Security implications considered
- [ ] Performance impact assessed
- [ ] Breaking changes documented

### Merge Requirements

- [ ] CI passes completely
- [ ] At least one approving review
- [ ] No conflicts with main branch
- [ ] Conventional commit title
- [ ] Documentation builds successfully

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [rustfmt Configuration](https://rust-lang.github.io/rustfmt/)
- [Clippy Documentation](https://rust-lang.github.io/rust-clippy/)