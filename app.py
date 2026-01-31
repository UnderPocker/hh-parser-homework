#!/usr/bin/env python
"""
Единый CLI: пайплайн по hh.csv или предсказание зарплат по x_data.npy.

Использование:
    python app.py path/to/hh.csv          — пайплайн: создаёт x_data.npy и y_data.npy рядом
    python app.py path/to/x_data.npy      — предсказание: выводит список зарплат (float) в JSON
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def run_pipeline(csv_path: Path) -> None:
    """Запуск пайплайна по hh.csv: создаёт x_data.npy и y_data.npy рядом с файлом."""
    from pipeline import DataPipeline

    if not csv_path.exists():
        logger.error("Файл не найден: %s", csv_path)
        sys.exit(1)
    if not csv_path.is_file():
        logger.error("Указанный путь не является файлом: %s", csv_path)
        sys.exit(1)

    output_dir = csv_path.parent
    logger.info("Входной файл: %s", csv_path)
    logger.info("Директория для сохранения результатов: %s", output_dir)

    try:
        pipeline = DataPipeline(output_dir=str(output_dir))
        pipeline.process(str(csv_path))
        x_path = output_dir / "x_data.npy"
        y_path = output_dir / "y_data.npy"
        print("\n[OK] Обработка завершена успешно!", file=sys.stdout)
        print("[OK] Созданы файлы:", file=sys.stdout)
        print(f"  - {x_path}", file=sys.stdout)
        print(f"  - {y_path}", file=sys.stdout)
    except Exception as e:
        logger.error("Ошибка при обработке данных: %s", e, exc_info=True)
        sys.exit(1)


def run_predict(x_data_path: Path, model_path: Optional[Path] = None) -> None:
    """Загрузка модели из resources/ и предсказание зарплат по x_data.npy. В stdout — JSON список float."""
    from model import SalaryRegressor

    if not x_data_path.exists():
        logger.error("Файл не найден: %s", x_data_path)
        sys.exit(1)
    if not x_data_path.is_file():
        logger.error("Указанный путь не является файлом: %s", x_data_path)
        sys.exit(1)

    try:
        X = np.load(x_data_path, allow_pickle=False)
    except Exception as e:
        logger.error("Ошибка загрузки x_data.npy: %s", e)
        sys.exit(1)

    if X.ndim != 2:
        logger.error("Ожидается 2D массив в x_data.npy")
        sys.exit(1)

    regressor = SalaryRegressor()
    try:
        regressor.load(model_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    salaries: List[float] = [float(s) for s in regressor.predict(X).tolist()]
    print(json.dumps(salaries, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Пайплайн по hh.csv или предсказание зарплат по x_data.npy"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Путь к hh.csv (пайплайн) или к x_data.npy (предсказание)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Путь к модели для предсказания (по умолчанию resources/model.npz)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    model_path = Path(args.model) if args.model else None

    if path.suffix.lower() == ".csv":
        run_pipeline(path)
    elif path.suffix.lower() == ".npy" or path.name.lower() == "x_data.npy":
        run_predict(path, model_path)
    else:
        logger.error(
            "Укажите файл .csv (пайплайн) или .npy / x_data.npy (предсказание): %s",
            path,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
