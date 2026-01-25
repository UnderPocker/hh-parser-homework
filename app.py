#!/usr/bin/env python
"""
CLI интерфейс для обработки данных из hh.csv.

Использование:
    python app.py path/to/hh.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from pipeline import DataPipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Главная функция CLI интерфейса."""
    parser = argparse.ArgumentParser(
        description='Обработка данных из CSV файла с использованием пайплайна Chain of Responsibility'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Путь к CSV файлу для обработки'
    )
    
    args = parser.parse_args()
    
    # Проверка существования файла
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"Файл не найден: {csv_path}")
        sys.exit(1)
    
    if not csv_path.is_file():
        logger.error(f"Указанный путь не является файлом: {csv_path}")
        sys.exit(1)
    
    # Определяем директорию для сохранения результатов (рядом с входным файлом)
    output_dir = csv_path.parent
    
    logger.info(f"Входной файл: {csv_path}")
    logger.info(f"Директория для сохранения результатов: {output_dir}")
    
    try:
        # Создаем и запускаем пайплайн
        pipeline = DataPipeline(output_dir=str(output_dir))
        pipeline.process(str(csv_path))
        
        x_path = output_dir / 'x_data.npy'
        y_path = output_dir / 'y_data.npy'
        
        print(f"\n[OK] Обработка завершена успешно!")
        print(f"[OK] Созданы файлы:")
        print(f"  - {x_path}")
        print(f"  - {y_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке данных: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
