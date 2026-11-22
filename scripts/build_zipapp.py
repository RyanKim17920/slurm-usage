#!/usr/bin/env python3
"""Build a single-file zipapp: ``python3 scripts/build_zipapp.py [-o dist/slurm-usage.pyz]``.

The result runs with any Python 3.8+ (``python3 slurm-usage.pyz``, or directly
when executable) and needs nothing else installed.
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipapp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "slurm_usage"


def build(output: str, interpreter: str = "/usr/bin/env python3") -> str:
    with tempfile.TemporaryDirectory() as staging:
        shutil.copytree(os.path.join(ROOT, PACKAGE), os.path.join(staging, PACKAGE),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        with open(os.path.join(staging, "__main__.py"), "w") as handle:
            handle.write("import sys\nfrom slurm_usage.cli import main\nsys.exit(main())\n")
        directory = os.path.dirname(os.path.abspath(output))
        os.makedirs(directory, exist_ok=True)
        zipapp.create_archive(staging, output, interpreter=interpreter, compressed=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default=os.path.join(ROOT, "dist", "slurm-usage.pyz"))
    args = parser.parse_args()
    print(build(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
