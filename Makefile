PYTHON ?= python3
PIP ?= pip3

export PYTHONPATH := src

setup:
	$(PIP) install -r requirements.txt

baseline:
	$(PYTHON) scripts/run_baseline_eval.py

train_short:
	$(PYTHON) scripts/run_micro_cpt.py --mode short

train_replay:
	$(PYTHON) scripts/run_micro_cpt.py --mode replay

eval:
	$(PYTHON) scripts/run_eval.py

report:
	$(PYTHON) scripts/make_report.py

ci: baseline train_short train_replay eval report
