PYTHON ?= python3

setup:
	$(PYTHON) -m pip install --upgrade pip
	# CI/runtime-saving choice: force CPU wheels to avoid multi-GB CUDA downloads on GitHub runners.
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.6"
	$(PYTHON) -m pip install -r requirements.txt

baseline:
	$(PYTHON) scripts/run_baseline_eval.py

train_short:
	$(PYTHON) scripts/run_micro_cpt.py --mode short

eval_short:
	$(PYTHON) scripts/run_eval.py --mode short

train_replay:
	$(PYTHON) scripts/run_micro_cpt.py --mode replay

eval:
	$(PYTHON) scripts/run_eval.py --mode replay

report:
	$(PYTHON) scripts/make_report.py

ci: baseline train_short eval_short train_replay eval report
