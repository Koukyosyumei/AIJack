jupyter-book build docs
sphinx-apidoc -f -o ./docs/api/source ./src/aijack
sphinx-build -b html ./docs/api/source ./docs/_build/html/api
ghp-import -n -p -f docs/_build/html
