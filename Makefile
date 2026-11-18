# Reproduce the analysis end to end:  make env && make all
PY ?= python
WORKERS ?= 4

.PHONY: env lint test fetch covariates extract members analyze figures all clean-results

env:
	conda env create -f environment.yml

lint:
	ruff check src tests

test:
	$(PY) -m pytest

fetch:                      ## about 30 GB from the public WeatherBench 2 bucket
	$(PY) -m landmem fetch metadata era5 forecasts truth members --workers $(WORKERS)

covariates:                 ## ERA5 only: domain, soil-moisture anomalies, coupling regime
	$(PY) -m landmem covariates

extract: covariates         ## afternoon/night forecast and truth panels
	$(PY) -m landmem extract

members: covariates         ## lead-day-5 ensemble scores for S7-S9
	$(PY) -m landmem members

analyze:                    ## registered tests -> results/results.json and results/tables
	$(PY) -m landmem analyze

figures:
	$(PY) -m landmem figures

all: fetch extract members analyze figures

clean-results:
	rm -f results/results.json results/tables/*.csv results/figures/*
