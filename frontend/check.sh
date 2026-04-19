#!/usr/bin/env bash
# Run all frontend quality checks
set -e
cd "$(dirname "$0")"
npm run check
