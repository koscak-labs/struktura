FROM rust:1.80-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin struktura

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/struktura /usr/local/bin/
ENTRYPOINT ["struktura"]
CMD ["--help"]
