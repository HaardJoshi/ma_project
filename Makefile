.PHONY: setup combine clean-data preprocess train evaluate all help

PYTHON   ?= python3
CONFIG   ?= configs/default.yaml

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv & install dependencies
	$(PYTHON) -m venv env
	env/bin/pip install --upgrade pip
	env/bin/pip install -r requirements.txt
	@echo "✅  Run 'source env/bin/activate' to activate."

combine:  ## Combine raw CSV exports → data/interim/ma_combined.csv
	$(PYTHON) scripts/run_combine.py

clean-data:  ## Clean combined CSV → data/interim/ma_cleaned.csv
	$(PYTHON) scripts/run_cleaning.py

preprocess:  ## Winsorise, z-score, split → data/processed/
	$(PYTHON) scripts/run_preprocessing.py --config $(CONFIG)

edgar-fetch:  ## Fetch 10-K filings from SEC EDGAR (set EMAIL=)
	$(PYTHON) scripts/run_edgar_fetch.py --email "$(EMAIL)"

train:  ## Train a model (use CONFIG=configs/xxx.yaml)
	$(PYTHON) train.py --config $(CONFIG)

evaluate:  ## Evaluate a trained model
	$(PYTHON) evaluate.py --config $(CONFIG)

all: combine clean-data preprocess train evaluate  ## Run full pipeline
