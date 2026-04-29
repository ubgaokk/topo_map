#!/bin/bash
# Run all tests with pytest (if available)

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Running TopoMap Unit Tests"
echo "=========================================="

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "Using pytest..."
    pytest tests/ -v --tb=short
else
    echo "Using unittest..."
    python tests/run_tests.py
fi

echo ""
echo "=========================================="
echo "Tests Complete"
echo "=========================================="