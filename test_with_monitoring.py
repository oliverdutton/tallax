#!/usr/bin/env python3
"""Run tests while monitoring system resources to identify the cause."""

import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output
import sys
import psutil
import os
import gc

def get_memory_info():
    """Get current process memory usage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        'rss_mb': mem.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem.vms / 1024 / 1024,  # Virtual Memory Size
    }

def get_system_memory():
    """Get system-wide memory info."""
    mem = psutil.virtual_memory()
    return {
        'total_mb': mem.total / 1024 / 1024,
        'available_mb': mem.available / 1024 / 1024,
        'percent': mem.percent,
    }

def run_monitored_tests():
    """Run tests while monitoring resources."""
    variants = [
        ("standard", {"return_argsort": False, "is_stable": False, "descending": False}),
        ("return_argsort", {"return_argsort": True, "is_stable": False, "descending": False}),
        ("stable", {"return_argsort": False, "is_stable": True, "descending": False}),
        ("stable_argsort", {"return_argsort": True, "is_stable": True, "descending": False}),
        ("descending", {"return_argsort": False, "is_stable": False, "descending": True}),
        ("descending_argsort", {"return_argsort": True, "is_stable": False, "descending": True}),
        ("descending_stable", {"return_argsort": False, "is_stable": True, "descending": True}),
        ("descending_stable_argsort", {"return_argsort": True, "is_stable": True, "descending": True}),
    ]

    dtypes = [jnp.bfloat16, jnp.float32]
    sizes = [128, 256]

    print("="*80)
    print("MONITORED TEST RUN")
    print("="*80)

    # Initial state
    initial_mem = get_memory_info()
    initial_sys = get_system_memory()
    print(f"Initial process memory: {initial_mem['rss_mb']:.1f} MB RSS, {initial_mem['vms_mb']:.1f} MB VMS")
    print(f"System memory: {initial_sys['available_mb']:.1f} MB available ({initial_sys['percent']:.1f}% used)")
    print("="*80)

    test_count = 0
    memory_stats = []

    for size in sizes:
        for dtype in dtypes:
            for variant_name, kwargs in variants:
                test_count += 1
                shape = (16, size)

                # Get memory before test
                mem_before = get_memory_info()
                sys_before = get_system_memory()

                test_name = f"{dtype.__name__}-{size}-{variant_name}"
                print(f"\n[{test_count:3d}] {test_name}")
                print(f"      Pre:  Process={mem_before['rss_mb']:7.1f}MB  System={sys_before['available_mb']:7.1f}MB avail", end="")
                sys.stdout.flush()

                key = jax.random.key(0)
                arr = jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype)
                operands = [arr]

                try:
                    verify_sort_output(
                        operands,
                        num_keys=1,
                        interpret=True,
                        **kwargs
                    )

                    # Get memory after test
                    mem_after = get_memory_info()
                    sys_after = get_system_memory()

                    delta_rss = mem_after['rss_mb'] - mem_before['rss_mb']
                    delta_sys = sys_before['available_mb'] - sys_after['available_mb']

                    print(f"\n      Post: Process={mem_after['rss_mb']:7.1f}MB  System={sys_after['available_mb']:7.1f}MB avail")
                    print(f"      Δ:    Process={delta_rss:+7.1f}MB  System={delta_sys:+7.1f}MB consumed")
                    print(f"      ✓ PASSED")

                    memory_stats.append({
                        'test': test_count,
                        'name': test_name,
                        'rss_mb': mem_after['rss_mb'],
                        'delta_rss_mb': delta_rss,
                        'sys_avail_mb': sys_after['available_mb'],
                        'delta_sys_mb': delta_sys,
                    })

                except Exception as e:
                    print(f"\n      ✗ FAILED: {type(e).__name__}")
                    print(f"\n💥 Test failed at position {test_count}")

                    # Get final memory
                    mem_final = get_memory_info()
                    sys_final = get_system_memory()

                    print(f"\nFinal process memory: {mem_final['rss_mb']:.1f} MB")
                    print(f"Final system memory: {sys_final['available_mb']:.1f} MB available")
                    print(f"Total process memory growth: {mem_final['rss_mb'] - initial_mem['rss_mb']:.1f} MB")
                    print(f"Total system memory consumed: {initial_sys['available_mb'] - sys_final['available_mb']:.1f} MB")

                    return test_count, test_name, memory_stats

                # Force garbage collection
                gc.collect()

    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)

    final_mem = get_memory_info()
    final_sys = get_system_memory()

    print(f"\nFinal process memory: {final_mem['rss_mb']:.1f} MB")
    print(f"Total process memory growth: {final_mem['rss_mb'] - initial_mem['rss_mb']:.1f} MB")
    print(f"\nFinal system memory: {final_sys['available_mb']:.1f} MB available")
    print(f"Total system memory consumed: {initial_sys['available_mb'] - final_sys['available_mb']:.1f} MB")

    return None, None, memory_stats


def analyze_memory_stats(stats, failed_at=None):
    """Analyze memory growth patterns."""
    if not stats:
        return

    print("\n" + "="*80)
    print("MEMORY ANALYSIS")
    print("="*80)

    # Find tests with largest memory growth
    sorted_by_delta = sorted(stats, key=lambda x: x['delta_rss_mb'], reverse=True)

    print("\nTop 5 tests by process memory growth:")
    for stat in sorted_by_delta[:5]:
        print(f"  [{stat['test']:2d}] {stat['name']:40s} +{stat['delta_rss_mb']:6.1f} MB")

    print("\nTop 5 tests by system memory consumption:")
    sorted_by_sys = sorted(stats, key=lambda x: x['delta_sys_mb'], reverse=True)
    for stat in sorted_by_sys[:5]:
        print(f"  [{stat['test']:2d}] {stat['name']:40s} +{stat['delta_sys_mb']:6.1f} MB")

    # Check for memory leak pattern
    print("\n" + "="*80)
    print("MEMORY LEAK DETECTION")
    print("="*80)

    first_half = stats[:len(stats)//2]
    second_half = stats[len(stats)//2:]

    avg_first = sum(s['delta_rss_mb'] for s in first_half) / len(first_half)
    avg_second = sum(s['delta_rss_mb'] for s in second_half) / len(second_half)

    print(f"Average memory growth (first half):  {avg_first:6.2f} MB/test")
    print(f"Average memory growth (second half): {avg_second:6.2f} MB/test")

    if avg_second > avg_first * 1.5:
        print("\n⚠️  MEMORY LEAK DETECTED: Later tests consume more memory")
        print("    This suggests accumulation of compilation artifacts or cached data")
    elif avg_second < avg_first * 0.5:
        print("\n✓ Memory usage decreases over time (possibly GC working)")
    else:
        print("\n→ Memory usage relatively stable")

    # Cumulative memory
    print("\n" + "="*80)
    print("CUMULATIVE MEMORY GROWTH")
    print("="*80)

    cumulative = 0
    for i, stat in enumerate(stats, 1):
        cumulative += stat['delta_rss_mb']
        if i % 5 == 0 or i == len(stats):
            print(f"  After test {i:2d}: {cumulative:+7.1f} MB total growth, {stat['sys_avail_mb']:7.1f} MB system available")

    if failed_at:
        print(f"\n⚠️  Test failed at position {failed_at}")
        print(f"    Cumulative memory before failure: {cumulative:+.1f} MB")


if __name__ == "__main__":
    print("Starting monitored test run...")
    print("This will track memory usage for each test\n")

    failed_pos, failed_test, stats = run_monitored_tests()

    if stats:
        analyze_memory_stats(stats, failed_pos)

    if failed_pos:
        print(f"\n{'='*80}")
        print(f"SEGFAULT at position {failed_pos}: {failed_test}")
        print('='*80)
