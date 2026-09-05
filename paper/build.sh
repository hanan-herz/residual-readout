#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -B ../scripts/build_descriptive_tables.py --check
python3 -B ../scripts/build_descriptive_figures.py
lualatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
lualatex -interaction=nonstopmode -halt-on-error paper.tex
lualatex -interaction=nonstopmode -halt-on-error paper.tex
