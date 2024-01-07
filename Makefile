include .env
export

# PYTEST_OPT ?= --cov=portrait_search

test:
	poetry run pytest ${PYTEST_OPT} tests/


tf-atlas:
	cd infra/atlas && terraform plan && terraform apply -auto-approve
