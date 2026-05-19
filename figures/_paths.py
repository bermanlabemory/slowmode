"""Common path setup for figure scripts.

Each figure script begins with `from _paths import *` and then loads data
relative to DATA_DIR and saves outputs to OUT_DIR.  The path layout is::

    slowmode/
        data/{lorenz,worms,flies}/...
        figures/<this directory>
        outputs/                       <-- created on first save
        figures.py                     <-- shared styling (setup_style, etc.)
"""
import os
import sys

# Repo root is one level above this file's directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, 'data')
LORENZ_DATA = os.path.join(DATA_ROOT, 'lorenz')
WORMS_DATA = os.path.join(DATA_ROOT, 'worms')
FLIES_DATA = os.path.join(DATA_ROOT, 'flies')
OUT_DIR = os.path.join(REPO_ROOT, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# Make the repo root importable so `from figures import ...` works.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from figures import setup_style as setup  # noqa: E402
from figures import ARM_PALETTE  # noqa: E402

ARM_PALETTE_2 = ['#0072B2', '#D55E00']  # worm Pirouette = blue, Run = red

ARM_TITLES_FLIES = [
    'Arm 1 — Idle & Slow',
    'Arm 2 — Anterior Movements',
    'Arm 3 — Posterior & Wing Movements',
    'Arm 4 — Locomotion',
]

setup()


def save(fig, name):
    """Save fig as both PNG and PDF to OUT_DIR."""
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=260, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    print(f'Saved outputs/{name}.{{png,pdf}}')
