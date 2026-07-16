.PHONY: setup lint test smoke-train train challenger-eval promote mlflow-ui clean

setup:
	pip install -r requirements.txt
	pip install -e .

lint:
	ruff check src tests

test:
	pytest tests -v

smoke-train:
	bash scripts/run_smoke_train.sh

train:
	python -m marl_packing.training.train --packer-config configs/train_packer.yaml \
		--physics-config configs/train_physics.yaml --env-config configs/env.yaml --full

challenger-eval:
	bash scripts/run_challenger_eval.sh

promote:
	python -m marl_packing.mlops.deploy --challenger $(CHALLENGER) --config configs/challenger_eval.yaml --auto-promote

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns

clean:
	rm -rf mlruns mlartifacts .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
