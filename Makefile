PYTHONPATH=src
PY=PYTHONPATH=$(PYTHONPATH) python

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

baseline:
	$(PY) scripts/run_baseline_eval.py

train_short:
	$(PY) scripts/run_micro_cpt.py --mode short

train_replay:
	$(PY) scripts/run_micro_cpt.py --mode replay

eval_short:
	$(PY) scripts/run_eval.py --checkpoint short

eval_replay:
	$(PY) scripts/run_eval.py --checkpoint replay

eval: eval_short eval_replay

report:
	$(PY) scripts/make_report.py

ci: baseline train_short eval_short train_replay eval_replay report
