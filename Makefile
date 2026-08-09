PYTHON ?= python
NPM ?= npm

.DEFAULT_GOAL := help

.PHONY: help install install-backend install-frontend install-ppo backend frontend test test-backend test-frontend lint build train-nominal train-robust benchmark docker-up docker-down

help:
	@$(PYTHON) -c "print('FIREOPS-SIM targets:\n  install         Install backend test and frontend dependencies\n  install-ppo     Install backend, test, and optional PPO dependencies\n  backend         Run the FastAPI development server\n  frontend        Run the Vite development server\n  test            Run backend tests plus frontend lint/build\n  train-nominal   Train a PPO policy at full sensor reliability\n  train-robust    Train a domain-randomized PPO policy\n  benchmark       Evaluate random and rule policies\n  docker-up       Build and run the full stack\n  docker-down     Stop the Compose stack')"

install: install-backend install-frontend

install-backend:
	$(PYTHON) -m pip install -e "./backend[test]"

install-frontend:
	$(NPM) --prefix frontend ci

install-ppo:
	$(PYTHON) -m pip install -e "./backend[test,ppo]"

backend:
	cd backend && $(PYTHON) -m uvicorn app.main:app --reload

frontend:
	$(NPM) --prefix frontend run dev

test: test-backend test-frontend

test-backend:
	cd backend && $(PYTHON) -m pytest

test-frontend:
	$(NPM) --prefix frontend run lint
	$(NPM) --prefix frontend run build

lint:
	$(NPM) --prefix frontend run lint

build:
	$(NPM) --prefix frontend run build

train-nominal:
	$(PYTHON) experiments/train.py --mode nominal --output models/ppo_nominal

train-robust:
	$(PYTHON) experiments/train.py --mode robust --output models/ppo_robust

benchmark:
	$(PYTHON) experiments/benchmark.py --output-dir results

docker-up:
	docker compose up --build

docker-down:
	docker compose down
