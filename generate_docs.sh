sphinx-apidoc -f -o ./docs/source ./src/aijack
sphinx-build -b html ./docs/source ./docs/_build/html
ghp-import -n -p -f docs/_build/html
