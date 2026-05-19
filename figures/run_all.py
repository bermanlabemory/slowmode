"""Run every figure script in order.

Generates every panel of the published manuscript (Kaur, Jain, Berman,
PRX Life 2026) into the repo's ``outputs/`` directory.  Each individual
script can also be run on its own, e.g.::

    python figures/figure_3.py

Estimated total runtime on a 2024 laptop without GPU: ~5--10 minutes.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# In paper-figure order.
SCRIPTS = [
    # ----- Main figures -----
    # Figure 2 (Lorenz) -- Panel A is generated inside lorenz.ipynb
    # (simulation-based); panels B--E are assembled in Illustrator from
    # the per-panel outputs produced by the scripts below.
    'figure_2B.py',
    'figure_2C.py',
    'figure_2D.py',
    'figure_2E.py',
    'figure_3.py',
    'figure_4.py',
    'figure_5.py',
    # ----- Supplementary figures -----
    'supp_figure_1.py',
    'supp_figure_2.py',
    'supp_figure_3.py',
    'supp_figure_4.py',
    'supp_figure_5.py',
    'supp_figure_6.py',
    'supp_figure_7.py',
    'supp_figure_8.py',
    'supp_figure_9.py',
    'supp_figure_10.py',
    'supp_figure_11.py',
]


def main():
    failed = []
    t0 = time.time()
    for script in SCRIPTS:
        path = os.path.join(HERE, script)
        print(f'\n=== {script} ===')
        result = subprocess.run([sys.executable, path], cwd=HERE)
        if result.returncode != 0:
            failed.append(script)
    elapsed = time.time() - t0
    print(f'\n--- Done in {elapsed:.1f} s ---')
    if failed:
        print(f'FAILED ({len(failed)}): {failed}')
        sys.exit(1)
    else:
        print(f'All {len(SCRIPTS)} scripts ran. See ../outputs/ for PNGs/PDFs.')


if __name__ == '__main__':
    main()
