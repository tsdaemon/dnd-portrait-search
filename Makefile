include .env
export

test:
	poetry run pytest --cov=portrait_search tests/
