dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

check:
	isort -c accurate_bg/
	black accurate_bg/ --check
	flake8 accurate_bg/
	flake8 time-gan/

format:
	isort -rc accurate_bg/
	black accurate_bg/
	flake8 accurate_bg/
	flake8 time-gan/
	black sub_job.py
	flake8 sub_job.py

.PHONY: dev check
