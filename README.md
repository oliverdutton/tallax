# tallax

`tallax` provides high-performance sorting and top-k operations for JAX, optimized for TPUs using Pallas.

## Installation

The installation process depends on your JAX backend (CPU or TPU).

### CPU Installation

For a CPU-only environment, you can install the package and its dependencies directly with pip:

```bash
pip install .
```

### TPU Installation

To use `tallax` on a TPU, you must first install the appropriate nightly JAX release.

1.  **Install the nightly JAX TPU wheel:**

    ```bash
    pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
    ```

2.  **Install `tallax`:**

    ```bash
    pip install .
    ```
