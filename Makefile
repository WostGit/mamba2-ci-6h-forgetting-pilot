PYTHON ?= python3

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

baseline:
	$(PYTHON) scripts/run_baseline_eval.py

train_short:
	$(PYTHON) scripts/run_micro_cpt.py --condition short

train_replay:
	$(PYTHON) scripts/run_micro_cpt.py --condition replay

eval:
	$(PYTHON) scripts/run_eval.py

report:
	$(PYTHON) scripts/make_report.py

ci:
	$(PYTHON) -c "from src.config import load_config; from src.preflight import assert_budget_feasible; assert_budget_feasible(load_config())"
	$(MAKE) baseline
	$(MAKE) train_short
	$(MAKE) train_replay
	$(MAKE) eval
	$(MAKE) report
