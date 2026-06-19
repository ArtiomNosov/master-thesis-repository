#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex
biber build/master-thesis-pz-body
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex
