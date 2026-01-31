#!/usr/bin/env python
"""
Запуск пайплайна обработки hh.csv (создаёт x_data.npy и y_data.npy рядом с файлом).

Использование:
    python run_pipeline.py path/to/hh.csv
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Запуск пайплайна по пути к hh.csv."""
    parser = argparse.ArgumentParser(
        description="Обработка данных из CSV (пайплайн Chain of Responsibility)"
    )
    parser.add_argument("csv_path", type=str, help="Путь к CSV файлу")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error("Файл не найден: %s", csv_path)
        sys.exit(1)
    if not csv_path.is_file():
        logger.error("Указанный путь не является файлом: %s", csv_path)
        sys.exit(1)

    output_dir = csv_path.parent
    logger.info("Входной файл: %s", csv_path)
    logger.info("Директория для сохранения: %s", output_dir)

    try:
        pipeline = DataPipeline(output_dir=str(output_dir))
        pipeline.process(str(csv_path))
        x_path = output_dir / "x_data.npy"
        y_path = output_dir / "y_data.npy"
        print("\n[OK] Обработка завершена успешно!")
        print("[OK] Созданы файлы:")
        print(f"  - {x_path}")
        print(f"  - {y_path}")
    except Exception as e:
        logger.error("Ошибка при обработке данных: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
