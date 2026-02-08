#!/usr/bin/env python
import mlflow

print(f'MLflow version: {mlflow.__version__}')
print(f'Has set_experiment_by_id: {hasattr(mlflow, "set_experiment_by_id")}')
methods = [m for m in dir(mlflow) if "experiment" in m.lower()]
print(f'Available experiment methods: {methods}')
