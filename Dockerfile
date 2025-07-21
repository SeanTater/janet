# Dockerfile for debugging CI test failures
FROM ubuntu:22.04

# Install dependencies similar to CI environment
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy workspace files
COPY . .

# Set environment variables similar to CI
ENV CARGO_TERM_COLOR=always
ENV RUSTFLAGS="-Dwarnings"
ENV CARGO_INCREMENTAL=0

# Run the specific test that's failing
CMD ["cargo", "test", "test_mcp_initialize", "--verbose", "--", "--nocapture"]
