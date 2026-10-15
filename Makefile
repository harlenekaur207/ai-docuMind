.PHONY: install dev api ui test

install:
	python -m pip install -r requirements-dev.txt

api:
	uvicorn app.main:app --reload

ui:
	streamlit run streamlit_app.py

test:
	pytest -q

