#!/usr/bin/env bash
# Kept for compatibility: the pipeline now lives in scripts/run.sh (any base model, quick mode).
exec bash "$(dirname "$0")/run.sh" "$@"
