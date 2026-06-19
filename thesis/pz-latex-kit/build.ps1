$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
docker run --rm -v "${PWD}:/work" -w /work aergus/latex:2022-01-02 bash -c @"
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex &&
biber build/master-thesis-pz-body &&
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex &&
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex
"@
