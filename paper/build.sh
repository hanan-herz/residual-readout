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

# The manuscript is expected to compile clean. -halt-on-error catches errors
# but not warnings, and a warning is how an undefined cross-reference, a
# dropped character, or an overfull box has slipped past before.
if grep -nE 'Warning:|warning \(|Overfull|Underfull|Missing character' paper.log; then
  echo "build.sh: paper.log has the warnings listed above; not accepting the PDF" >&2
  exit 1
fi
echo "build.sh: paper.pdf built with no warning lines"
