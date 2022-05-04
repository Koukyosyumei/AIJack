pandoc -f markdown -t rst -o docs_src/README.rst README.md
sphinx-apidoc -M -f -o ./docs_src ./src/aijack
sphinx-build -b html docs_src docs
