#!/bin/bash

echo "🛠️  Building Docker image..."
docker build -f docker/prep_features -t etl-pipeline . || { echo "❌ Build failed"; exit 1; }

echo "🚀 Running container..."
docker run --rm -v /mnt/data:/mnt/data etl-pipeline || { echo "❌ Container run failed"; exit 1; }

echo "✅ Done. Output should be at /mnt/data/processed/train_ready.csv"
