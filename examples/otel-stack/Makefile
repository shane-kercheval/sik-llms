# OpenTelemetry Stack Management
# Based on examples/otel_101.md "Complete Stack: Traces + Metrics (Jaeger + Prometheus Example)"

.PHONY: help start stop status logs clean clean-all clean-traces clean-metrics clean-dashboards data-info verify test-telemetry

help: ## Show this help message
	@echo "OpenTelemetry Complete Observability Stack"
	@echo "========================================"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

start: ## Start the complete observability stack
	@echo "🚀 Starting OpenTelemetry observability stack..."
	docker compose up -d
	@echo ""
	@echo " Stack started! Access points:"
	@echo "   =� Jaeger (Traces):     http://localhost:16686"
	@echo "   =� Prometheus (Metrics): http://localhost:9090"
	@echo "   =� Grafana (Dashboards): http://localhost:3000 (admin/admin)"
	@echo ""
	@echo "=' To enable sik-llms telemetry:"
	@echo "   export OTEL_SDK_DISABLED=false"
	@echo "   export OTEL_SERVICE_NAME=\"my-llm-app\""
	@echo "   export OTEL_EXPORTER_OTLP_ENDPOINT=\"http://localhost:4318/v1/traces\""

stop: ## Stop the observability stack
	@echo "🛑 Stopping OpenTelemetry observability stack..."
	docker compose down

status: ## Show status of all services
	@echo "=� OpenTelemetry Stack Status:"
	@echo "=============================="
	docker compose ps

logs: ## Show logs from all services
	docker compose logs -f

logs-jaeger: ## Show logs from Jaeger only
	docker compose logs -f jaeger

logs-prometheus: ## Show logs from Prometheus only
	docker compose logs -f prometheus

logs-collector: ## Show logs from OTLP Collector only
	docker compose logs -f otel-collector

logs-grafana: ## Show logs from Grafana only
	docker compose logs -f grafana

clean: ## Stop and remove containers (keeps data volumes)
	@echo ">� Cleaning up OpenTelemetry stack..."
	docker compose down --remove-orphans
	@echo " Cleanup complete"

verify: ## Verify all services are healthy and accessible
	@echo "Verifying OpenTelemetry stack health..."
	@echo ""
	@echo "Checking Jaeger UI..."
	@curl -s -o /dev/null -w "  Jaeger UI: %{http_code}\n" http://localhost:16686/ || echo "  Jaeger UI: L Not accessible"
	@echo ""
	@echo "Checking Jaeger OTLP endpoint..."
	@curl -s -o /dev/null -w "  Jaeger OTLP: %{http_code}\n" -X POST -H "Content-Type: application/json" -d '{"resourceSpans":[]}' http://localhost:4318/v1/traces || echo "  Jaeger OTLP: L Not accessible"
	@echo ""
	@echo "Checking Prometheus UI..."
	@curl -s -o /dev/null -w "  Prometheus UI: %{http_code}\n" http://localhost:9090/ || echo "  Prometheus UI: L Not accessible"
	@echo ""
	@echo "Checking Prometheus API..."
	@curl -s -o /dev/null -w "  Prometheus API: %{http_code}\n" http://localhost:9090/api/v1/query?query=up || echo "  Prometheus API: L Not accessible"
	@echo ""
	@echo "Checking Grafana UI..."
	@curl -s -o /dev/null -w "  Grafana UI: %{http_code}\n" http://localhost:3000/ || echo "  Grafana UI: L Not accessible"
	@echo ""
	@echo "Checking OTLP Collector health..."
	@curl -s -o /dev/null -w "  OTLP Collector: %{http_code}\n" http://localhost:4318/v1/traces || echo "  OTLP Collector: L Not accessible"


test-telemetry: ## Test telemetry with comprehensive examples (requires sik-llms[telemetry])
	@echo ">� Testing telemetry integration..."
	@echo "Note: Run 'uv sync' first and set OpenAI API key"
	@echo ""
	OTEL_SDK_DISABLED=false \
	OTEL_SERVICE_NAME="otel-stack-test" \
	OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces" \
	uv run example_with_telemetry.py
	@echo ""
	@echo "🎯 Telemetry test complete! View traces at http://localhost:16686"

test-simple: ## Run simple telemetry test (quick verification)
	@echo ">� Running simple telemetry test..."
	@echo "Note: Run 'uv sync' first and set OpenAI API key"
	@echo ""
	OTEL_SDK_DISABLED=false \
	OTEL_SERVICE_NAME="otel-stack-simple" \
	OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces" \
	uv run test_simple.py

restart: ## Restart the entire stack
	@make stop
	@make start

# Development commands
dev-start: ## Start stack with immediate logs
	docker compose up

dev-rebuild: ## Rebuild and start (useful after config changes)
	docker compose up --build

# Maintenance commands  
pull: ## Pull latest images
	@echo "=� Pulling latest container images..."
	docker compose pull

backup-grafana: ## Backup Grafana data
	@echo "=� Backing up Grafana data..."
	docker run --rm -v otel-stack_grafana-storage:/source -v $(PWD):/backup alpine tar czf /backup/grafana-backup.tar.gz -C /source .
	@echo " Grafana backup saved as grafana-backup.tar.gz"

restore-grafana: ## Restore Grafana data from backup
	@echo "=� Restoring Grafana data..."
	@test -f grafana-backup.tar.gz || (echo "L grafana-backup.tar.gz not found" && exit 1)
	docker run --rm -v otel-stack_grafana-storage:/target -v $(PWD):/backup alpine tar xzf /backup/grafana-backup.tar.gz -C /target
	@echo " Grafana data restored"

clean-all: ## DESTROY ALL DATA - remove containers, networks, and volumes
	@echo "WARNING: This will permanently delete all traces, metrics, and dashboards\!"
	@read -p "Are you sure? [y/N] " -n 1 -r; echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v --remove-orphans; \
		echo "Complete cleanup finished"; \
	else \
		echo "Cleanup cancelled"; \
	fi

clean-traces: ## Clear only Jaeger traces (keeps metrics and dashboards)
	@echo "WARNING: This will permanently delete all Jaeger traces!"
	@read -p "Are you sure? [y/N] " -n 1 -r; echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose stop jaeger; \
		docker volume rm otel-stack_jaeger-badger 2>/dev/null || true; \
		docker compose up -d jaeger; \
		echo "✅ Jaeger traces cleared"; \
	else \
		echo "❌ Cleanup cancelled"; \
	fi

clean-metrics: ## Clear only Prometheus metrics (keeps traces and dashboards)
	@echo "WARNING: This will permanently delete all Prometheus metrics!"
	@read -p "Are you sure? [y/N] " -n 1 -r; echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose stop prometheus; \
		docker volume rm otel-stack_prometheus-data 2>/dev/null || true; \
		docker compose up -d prometheus; \
		echo "✅ Prometheus metrics cleared"; \
	else \
		echo "❌ Cleanup cancelled"; \
	fi

clean-dashboards: ## Clear only Grafana dashboards and settings (keeps traces and metrics)
	@echo "WARNING: This will permanently delete all Grafana dashboards and settings!"
	@read -p "Are you sure? [y/N] " -n 1 -r; echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose stop grafana; \
		docker volume rm otel-stack_grafana-storage 2>/dev/null || true; \
		docker compose up -d grafana; \
		echo "✅ Grafana dashboards and settings cleared"; \
	else \
		echo "❌ Cleanup cancelled"; \
	fi

data-info: ## Show information about data persistence
	@echo "Data Persistence Status:"
	@echo "======================="
	@echo "✅ Jaeger (Traces):     PERSISTED in jaeger-badger volume"
	@echo "✅ Prometheus (Metrics): PERSISTED in prometheus-data volume (30 day retention)"
	@echo "✅ Grafana (Dashboards): PERSISTED in grafana-storage volume"
	@echo ""
	@echo "Management Commands:"
	@echo "  make clean           - Stop containers but keep all data"
	@echo "  make clean-all       - DESTROY all data (requires confirmation)"
	@echo ""
	@echo "Selective Cleaning Commands:"
	@echo "  make clean-traces    - Clear only Jaeger traces"
	@echo "  make clean-metrics   - Clear only Prometheus metrics"  
	@echo "  make clean-dashboards - Clear only Grafana dashboards"
	@echo ""
	@echo "Volume Information:"
	@docker volume ls | grep otel-stack || echo "No volumes found (stack not started)"
	@echo ""
	@echo "💡 Tip: Use selective cleaning during development to clear specific data types"
	@echo "    while preserving others (e.g., clear traces but keep your Grafana dashboards)"
