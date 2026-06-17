"""Compatibility entrypoint for teacher evaluation.

Use ``src/training/teacher_test.py`` for new scripts. This file is kept so
older commands that call ``src/training/test.py`` run the same evaluator.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.teacher.evaluate import main


if __name__ == "__main__":
    main()
