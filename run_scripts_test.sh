export DISABLE_LATEX=1
pytest --cov amoc_collapse_scripts -v -n auto --nbmake notebooks/*.ipynb --durations 0 --ignore amoc_collapse_scripts/imports.py
