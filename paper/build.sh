#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Keep the tracked PDF stable across identical builds.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"
export FORCE_SOURCE_DATE=1

python3 -B ../scripts/build_descriptive_tables.py --check
python3 -B ../scripts/build_descriptive_figures.py
lualatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
lualatex -interaction=nonstopmode -halt-on-error paper.tex
lualatex -interaction=nonstopmode -halt-on-error paper.tex
