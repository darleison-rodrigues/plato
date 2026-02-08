.PHONY: setup start process search stats docker-up docker-down

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Checking Ollama..."
	ollama serve & > /dev/null 2>&1 || true
	@echo "Pulling models..."
	ollama pull llama3.2:3b
	ollama pull nomic-embed-text

start:
	@python main.py --help

process:
	@python main.py process documents/ --batch

search:
	@python main.py search "$(QUERY)"

stats:
	@python main.py stats

docker-up:
	docker compose up -d --build
	@echo "Services started. Run 'docker compose exec app python main.py --help' to use."

docker-down:
	docker compose down
