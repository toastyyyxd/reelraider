# ReelRaider Docker Management

.PHONY: help build up down logs clean dev prod restart

help: ## Show this help message
	@echo "ReelRaider Docker Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build all Docker images
	docker-compose build

up: ## Start services in production mode
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## View logs from all services
	docker-compose logs -f

logs-web: ## View logs from web service only
	docker-compose logs -f reelraider-web

logs-service: ## View logs from Python service only
	docker-compose logs -f reelraider-service

clean: ## Remove all containers, images, and volumes
	docker-compose down -v --rmi all

dev: ## Start services in development mode
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

dev-build: ## Build and start in development mode
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

prod: ## Start services in production mode (same as up)
	docker-compose up -d

restart: ## Restart all services
	docker-compose restart

restart-web: ## Restart web service only
	docker-compose restart reelraider-web

restart-service: ## Restart Python service only
	docker-compose restart reelraider-service

status: ## Show status of all services
	docker-compose ps

shell-web: ## Open shell in web container
	docker-compose exec reelraider-web sh

shell-service: ## Open shell in Python service container
	docker-compose exec reelraider-service bash

test-web: ## Test web service endpoint
	curl -f http://localhost:8000/ || echo "Web service not responding"

# Data processing commands
init-data: ## Initialize data processing (run in service container)
	docker-compose exec reelraider-service python -m datasets.fetch_raw
	docker-compose exec reelraider-service python -m datasets.ids
	docker-compose exec reelraider-service python -m datasets.ratings
	docker-compose exec reelraider-service python -m datasets.omdb

build-embeddings: ## Build embedding components
	docker-compose exec reelraider-service python -m datasets.culturally_aware_embedding

# Quick commands
quick-start: build up ## Build and start all services
quick-dev: dev-build ## Quick development start
quick-prod: build prod ## Quick production start
