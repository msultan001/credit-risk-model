.PHONY: install test train dashboard clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ --cov=src

train:
	python src/train.py

dashboard:
	streamlit run src/dashboard.py

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf models/*.pkl
