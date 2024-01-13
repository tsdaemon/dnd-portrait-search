include .env
export

# PYTEST_OPT ?= --cov=portrait_search

test:
	poetry run pytest ${PYTEST_OPT} tests/


validate-datasets:
	poetry run python -m portrait_search.validate_datasets


tf-atlas:
	cd infra/atlas && terraform plan && terraform apply -auto-approve
