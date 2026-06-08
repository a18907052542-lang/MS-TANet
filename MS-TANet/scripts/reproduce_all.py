"""
Master reproduction script.

Running this single entry point regenerates every CSV table and every PNG
figure in the `results/` directory.  Use this for a one-shot end-to-end
reproduction of the paper's main quantitative claims.
"""

import os
import sys
import time
import importlib

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TABLE_MODULES = [
    "scripts.reproduce_table3",
    "scripts.reproduce_table4",
    "scripts.reproduce_table5",
    "scripts.reproduce_table6",
    "scripts.reproduce_table7",
    "scripts.reproduce_table8",
    "scripts.reproduce_table9",
    "scripts.reproduce_table10",
    "scripts.reproduce_table11",
    "scripts.reproduce_table12",
]


def main():
    t0 = time.time()
    print("=" * 70)
    print("MS-TANet — full quantitative reproduction")
    print("=" * 70)

    print("\n[1/2]  Regenerating result tables in results/tables/")
    table_paths = []
    for mod_name in TABLE_MODULES:
        mod = importlib.import_module(mod_name)
        path = mod.reproduce()
        print(f"        ✓ {os.path.basename(path)}")
        table_paths.append(path)

    print("\n[2/2]  Regenerating publication figures in results/figures/")
    from visualization import FIGURE_RENDERERS
    figure_paths = []
    for name, fn in FIGURE_RENDERERS.items():
        path = fn()
        print(f"        ✓ {os.path.basename(path)}")
        figure_paths.append(path)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Reproduction complete in {elapsed:.1f} s.")
    print(f"  {len(table_paths)} tables  →  results/tables/")
    print(f"  {len(figure_paths)} figures →  results/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
