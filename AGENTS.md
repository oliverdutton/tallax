# Agent Guidelines

## Environment Setup

To set up the environment for testing and development:

1.  **Install Dependencies:**
    You need to install the package in editable mode. For CPU testing (default):
    ```bash
    pip install -e .
    ```
    For TPU support:
    ```bash
    pip install -e ".[tpu]"
    ```

    You also need `pytest` for running tests:
    ```bash
    pip install pytest
    ```

2.  **Running Tests:**
    Run the tests using `pytest`:
    ```bash
    pytest tests/test_sort_correctness.py
    ```

    Note: Correctness tests run in interpreter mode (`interpret=True`) on CPU by default to verify logic without requiring TPU hardware.
