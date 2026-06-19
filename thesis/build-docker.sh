#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
latex="$root/latex"
mount="${latex//\\//}"
docker run --rm -v "${mount}:/work" -w /work aergus/latex:2022-01-02 bash -c '
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex &&
biber build/master-thesis-pz-body &&
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex &&
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex
'
echo "PDF: ${latex}/build/master-thesis-pz-body.pdf"
