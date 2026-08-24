#!/bin/bash
# struktura DevOps integration examples
# add these to cron, systemd timers, or CI pipelines

# === 1. CRONTAB: check a metric every 5 minutes ===
# */5 * * * * /usr/local/bin/struktura guard /var/log/latency.csv --json >> /var/log/struktura.log 2>&1

# === 2. PROMETHEUS: pipe query results through DFA ===
# curl -s 'http://prometheus:9090/api/v1/query?query=rate(http_requests_total[5m])' \
#   | jq -r '.data.result[0].values[][1]' \
#   | struktura pipe --json

# === 3. SYSTEMD TIMER: continuous monitoring ===
# /etc/systemd/system/struktura-guard.service
# [Service]
# ExecStart=/usr/local/bin/struktura guard /var/metrics/cpu.csv --watch --webhook https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
# Restart=always

# === 4. CI PIPELINE: fail the build if structure degraded ===
# - name: Structure check
#   run: |
#     struktura check test-results/performance.csv
#     # exit code: 0=healthy, 1=fault detected, 2=error

# === 5. DOCKER: instant deployment ===
# docker run -v /your/data:/data struktura guard /data/sensor.csv --json

# === 6. PIPE FROM ANY SOURCE ===
# tail -f /var/log/sensor.csv | struktura pipe --window 128 --json
# mqtt sub sensor/temperature | struktura pipe
# kafka-console-consumer --topic metrics | struktura pipe

echo "struktura devops integration examples — see comments in this file"
echo "run: struktura guard <your-csv> --json"
