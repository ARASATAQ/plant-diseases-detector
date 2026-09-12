.PHONY: install weights run test clean

install:
	pip install -r requirements.txt

weights:
	python download_weights.py

run:
	python run.py

test:
	pytest tests/ -v

clean:
	for /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
	if exist .pytest_cache rd /s /q .pytest_cache
