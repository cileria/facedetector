.PHONY: help venv create-dirs install install-dev install-train install-all download-models verify-training-data \
        run run-custom lint format clean clean-all setup setup-dev setup-train setup-all check test test-timeout \
        notebook migrate-code reinstall setup-training-data

# Variables
PYTHON := python3.10
VENV := .venv
BIN := $(VENV)/bin
MODEL_DIR := model
OUTPUT_DIR := output
INPUT_DIR := input
TRAINING_DATA_DIR := videos/trainingsdata/jan

# Help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Core setup targets
$(VENV)/bin/activate: pyproject.toml  ## Create virtual environment
	@echo "Creating virtual environment..."
	uv venv --python $(PYTHON) $(VENV)
	. $(BIN)/activate
	touch $(BIN)/activate

venv: $(VENV)/bin/activate  ## Create virtual environment if it doesn't exist

create-dirs: ## Create necessary directories
	mkdir -p $(MODEL_DIR) $(OUTPUT_DIR) $(INPUT_DIR) src/video_tracking

# Installation targets
install: venv create-dirs ## Install package with core dependencies
	uv pip install -e .

install-dev: venv create-dirs ## Install package with development dependencies
	uv pip install -e ".[dev]"

install-train: venv create-dirs ## Install package with training dependencies
	uv pip install -e ".[train]"

install-all: venv create-dirs ## Install package with all dependencies
	uv pip install -e ".[all]"

# Model and training data management
download-models: ## Download required ML models
	@echo "Downloading YOLO model..."
	@if [ ! -f "$(MODEL_DIR)/yolo11n.pt" ]; then \
		$(BIN)/python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('$(MODEL_DIR)/yolo11n.pt')"; \
	fi

setup-training-data: ## Create and verify training data directory structure
	mkdir -p $(TRAINING_DATA_DIR)
	@echo "Training data directory: $(TRAINING_DATA_DIR)"
	@if [ -n "$$(ls -A $(TRAINING_DATA_DIR))" ]; then \
		echo "Found training images:"; \
		ls -la $(TRAINING_DATA_DIR); \
	else \
		echo "Please place your training images in: $(TRAINING_DATA_DIR)"; \
		echo "The directory structure should be:"; \
		echo "videos/"; \
		echo "└── trainingsdata/"; \
		echo "    └── jan/"; \
		echo "        ├── person1.jpg"; \
		echo "        ├── person2.jpg"; \
		echo "        └── ..."; \
	fi

verify-training-data: ## Verify training data exists
	@if [ -z "$$(find $(TRAINING_DATA_DIR) -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" \))" ]; then \
		echo "WARNING: No training images found in $(TRAINING_DATA_DIR)"; \
		echo "Please add training images before running the application"; \
		exit 1; \
	else \
		echo "Found training images:"; \
		ls -la $(TRAINING_DATA_DIR)/*.{jpg,png} 2>/dev/null || true; \
	fi

# Run targets
run: install verify-training-data ## Run the application with example videos
	$(BIN)/python -m video_tracking.main \
		"videos/test/JAN_VID_1.mp4" \
		"videos/test/NACHO_VID_1.mp4" \
		"videos/test/TESTWOMAN.mp4" \

run-custom: install verify-training-data ## Run with custom video (VIDEO_PATH required)
	@if [ -z "$(VIDEO_PATH)" ]; then \
		echo "Error: VIDEO_PATH is required. Usage: make run-custom VIDEO_PATH=path/to/video.mp4"; \
		exit 1; \
	fi
	$(BIN)/python -m video_tracking.main "$(VIDEO_PATH)"

# Development targets
lint: install-dev ## Run ruff and mypy
	$(BIN)/ruff check .
	$(BIN)/mypy .

format: install-dev ## Format code using ruff
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

test: install-dev ## Run tests
	$(BIN)/pytest tests/ -v --cov=src --cov-report=term-missing

test-timeout: install-dev ## Run tests with timeout
	$(BIN)/pytest tests/ -v --cov=src --cov-report=term-missing --timeout=300

check: lint ## Run all checks

notebook: install-all ## Start Jupyter notebook
	$(BIN)/jupyter notebook

# Setup targets
setup: install setup-training-data verify-training-data download-models ## Setup basic environment
	@echo "Environment ready!"

setup-dev: clean install-dev download-models ## Setup development environment
	@echo "Development environment ready!"

setup-train: clean install-train download-models ## Setup training environment
	@echo "Training environment ready!"

setup-all: clean install-all download-models ## Setup complete environment
	@echo "Complete environment ready!"

# Cleanup targets
clean: ## Clean up cache and build files
	rm -rf build/ dist/ *.egg-info .coverage .mypy_cache .pytest_cache .ruff_cache *.pt
	rm -rf $(OUTPUT_DIR)/* $(INPUT_DIR)/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(VENV)

clean-all: clean ## Clean everything including downloaded models
	rm -rf $(MODEL_DIR)/*

reinstall: clean-all setup run ## Reinstall everything from scratch

# Utility targets
migrate-code: create-dirs ## Migrate existing code to src structure
	@if [ -f "main.py" ] && [ ! -f "$(SRC_DIR)/main.py" ]; then \
		mv main.py $(SRC_DIR)/main.py; \
	fi