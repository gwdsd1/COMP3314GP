"""Score frozen DATER models with the controls.json reader corrected.

The frozen fitting driver remains unchanged. Its evaluate stage incorrectly
passes controls.json to a CSV reader; this adapter corrects only that reader.
"""
import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits
import run_orl_dater as D


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/orl_dater')
    args = parser.parse_args()
    D.ROOT = D.BASE / args.output
    original_csv_rows = D.csv_rows
    D.csv_rows = lambda path: D.read(path) if Path(path).suffix == '.json' else original_csv_rows(path)
    with threadpool_limits(limits=2):
        D.evaluate()
