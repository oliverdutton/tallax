# tallax

`tallax` provides high-performance sorting and top-k operations for JAX, optimized for TPUs using Pallas.

## Installation

The installation process for `tallax` depends on your JAX backend (CPU or TPU). For `tallax` to function correctly, you must first install the appropriate version of JAX for your hardware.

### 1. Install JAX

Follow the [official JAX installation guide](https://github.com/google/jax#installation) to install `jax` and `jaxlib` for your specific accelerator (CPU, GPU, or TPU).

For example, to install JAX for a TPU environment, you might run:
```bash
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 2. Install tallax

Once JAX is installed, you can install `tallax` using pip:

```bash
pip install .
```

If you installed `jax[tpu]`, the `tallax` installation will automatically use it. Otherwise, it will use the CPU version of JAX.
