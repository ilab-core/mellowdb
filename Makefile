# Install the project
setup:
	python setup.py install
	python -m pip install .[server]
	chmod +x /mellow-db/setup-env.sh
	/mellow-db/setup-env.sh
# Run tests
test:
	python -m pip install .[server,pytest]
	python -m pytest -k "not (test_concurrent_operations or test_back_up_and_load)"

# Clean up build artifacts and cache
clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -exec rm -rf {} +
	find . -name "*.pyo" -exec rm -rf {} +

help:
	@echo "Available targets:"
	@echo "  setup         - Install the project and server dependencies"
	@echo "  test          - Install test dependencies and run tests"
	@echo "  clean         - Remove build artifacts and Python caches"
	@echo "  help          - Display this help message"

.PHONY: setup test clean help
