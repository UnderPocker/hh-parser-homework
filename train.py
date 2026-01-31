#!/usr/bin/env python
"""
Обучение регрессионной модели на выходе пайплайна (x_data.npy, y_data.npy).
Веса сохраняются в папку resources репозитория.

Использование:
    python train.py path/to/data_dir
    python train.py path/to/x_data.npy path/to/y_data.npy
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from model import SalaryRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Обучение модели и сохранение весов в resources."""
    parser = argparse.ArgumentParser(
        description="Обучает регрессионную модель на выходе пайплайна и сохраняет веса в resources/"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=str,
        default=None,
        help="Директория с x_data.npy и y_data.npy",
    )
    parser.add_argument(
        "--x",
        type=str,
        default=None,
        help="Путь к x_data.npy (вместе с --y)",
    )
    parser.add_argument(
        "--y",
        type=str,
        default=None,
        help="Путь к y_data.npy (вместе с --x)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Коэффициент L2-регуляризации Ridge (по умолчанию 1.0)",
    )
    args = parser.parse_args()

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        x_path = data_dir / "x_data.npy"
        y_path = data_dir / "y_data.npy"
    elif args.x and args.y:
        x_path = Path(args.x)
        y_path = Path(args.y)
    else:
        parser.error("Укажите data_dir или оба --x и --y")

    for p, name in [(x_path, "x_data.npy"), (y_path, "y_data.npy")]:
        if not p.exists():
            logger.error("Файл не найден: %s", p)
            sys.exit(1)

    logger.info("Загрузка данных: %s, %s", x_path, y_path)
    X = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)

    if X.ndim != 2 or y.ndim != 1:
        logger.error("Ожидаются X (2d) и y (1d)")
        sys.exit(1)
    if X.shape[0] != y.shape[0]:
        logger.error("Число строк X и y не совпадает")
        sys.exit(1)

    logger.info("Размер выборки: %s, признаков: %s", X.shape[0], X.shape[1])

    regressor = SalaryRegressor(alpha=args.alpha)
    regressor.fit(X, y)
    regressor.save()

    logger.info("Обучение завершено. Веса сохранены в resources/")


if __name__ == "__main__":
    main()
