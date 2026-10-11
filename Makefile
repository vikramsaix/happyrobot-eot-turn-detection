PYTHON ?= python3

.PHONY: train test serve benchmark
train:
	$(PYTHON) train.py

test:
	$(PYTHON) -m unittest discover -s . -p 'test_*.py' -v

serve:
	$(PYTHON) -m uvicorn turn_detector.service:app --host 0.0.0.0 --port 8000

benchmark:
	$(PYTHON) bench.py --requests 1200 --concurrency 20
