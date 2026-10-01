PYTHON ?= python

POLL_INTERVAL_SECONDS ?= 15

.PHONY: install test foundation-smoke simulator preprocess train infer dashboard run stop

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

foundation-smoke:
	$(PYTHON) -c 'from pipeline.config import load_config; from pipeline.paths import ensure_data_dirs; config = load_config(); paths = ensure_data_dirs(); print("Config:", config); print("Ensured data directories:"); print("\n".join(f"  {name}: {path} (exists={path.is_dir()})" for name, path in paths.items()))'

simulator:
	$(PYTHON) -m pipeline.simulator

preprocess:
	$(PYTHON) -m pipeline.preprocess

train:
	$(PYTHON) -m pipeline.train

infer:
	$(PYTHON) -m pipeline.infer

dashboard:
	$(PYTHON) -m streamlit run pipeline/dashboard.py

run:
	POLL_INTERVAL_SECONDS=$(POLL_INTERVAL_SECONDS) $(PYTHON) -m pipeline.process_manager start

stop:
	$(PYTHON) -m pipeline.process_manager stop
