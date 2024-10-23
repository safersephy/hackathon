format:
	ruff format src


lint:
	ruff check src --fix
	mypy src --ignore-missing-imports

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
	
# `stop_server` target: Check if an MLflow server is running on port 5000 and shut it down if it is
# Find the process listening on port 5000, filter by 'mlflow', extract its process ID, and terminate it
stop_server:
	@-lsof -i :5000 -sTCP:LISTEN | grep 'mlflow' | awk '{ print $$2 }' | xargs