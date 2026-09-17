FROM registry.gitlab.hpi.de/akita/i/rust-base

LABEL maintainer="webfastonline@gmail.com"

ENV ALGORITHM_MAIN="/app/target/release/struktura-dfa-timeeval"

COPY Cargo.toml Cargo.lock ./
COPY src ./src
# The adapter depends on the struktura crate via path = "../.."
# In the Docker build context (run from the repo root), we need the crate source.
COPY ../../Cargo.toml /struktura/Cargo.toml
COPY ../../src /struktura/src

# Patch the path dependency to point at /struktura inside the container
RUN sed -i 's|path = "../.."|path = "/struktura"|' Cargo.toml && \
    cargo build --release
