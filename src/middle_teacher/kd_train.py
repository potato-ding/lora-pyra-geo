"""Certified single-pass entry; reuses R0 optimizer/selection/checkpoint loop."""
from src.middle_teacher.r0_train import main

if __name__ == '__main__':
    main(allow_kd=True)
