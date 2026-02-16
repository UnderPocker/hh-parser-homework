#!/usr/bin/env python
"""
PoC: Классификация IT-разработчиков по уровням (junior/middle/senior).

Использование:
    python classify.py path/to/hh.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from classifier import (
    ClassBalanceVisualizer,
    DeveloperLevelClassifier,
    ITDeveloperProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Главная функция для классификации IT-разработчиков."""
    parser = argparse.ArgumentParser(
        description="Классификация IT-разработчиков по уровням (junior/middle/senior)"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Путь к CSV файлу с данными hh.ru",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Директория для сохранения результатов (по умолчанию рядом с CSV)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля тестовой выборки (по умолчанию 0.2)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error("Файл не найден: %s", csv_path)
        sys.exit(1)

    # Определяем директорию для сохранения результатов
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "classification_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PoC: Классификация IT-разработчиков по уровням")
    logger.info("=" * 70)
    logger.info(f"Входной файл: {csv_path}")
    logger.info(f"Директория результатов: {output_dir}")

    try:
        # 1. Загрузка данных
        logger.info("\n[1/5] Загрузка данных...")
        df = pd.read_csv(csv_path, encoding='utf-8')
        logger.info(f"Загружено резюме: {len(df)}")

        # 2. Обработка данных
        logger.info("\n[2/5] Обработка данных...")
        processor = ITDeveloperProcessor()
        it_df, levels = processor.process_dataframe(df)

        if len(it_df) == 0:
            logger.error("Не найдено IT-разработчиков в датасете")
            sys.exit(1)

        logger.info(f"Обработано IT-резюме: {len(it_df)}")
        logger.info(f"Распределение уровней:\n{levels.value_counts()}")

        # Сохраняем обработанные данные
        processed_data_path = output_dir / "processed_data.csv"
        it_df_with_levels = it_df.copy()
        it_df_with_levels['level'] = levels
        it_df_with_levels.to_csv(processed_data_path, index=False, encoding='utf-8')
        logger.info(f"Обработанные данные сохранены: {processed_data_path}")

        # 3. Визуализация баланса классов
        logger.info("\n[3/5] Построение графиков баланса классов...")
        visualizer = ClassBalanceVisualizer(output_dir=output_dir)
        visualizer.plot_class_balance(
            levels,
            title="Распределение уровней IT-разработчиков",
        )
        visualizer.plot_class_distribution_detailed(levels)

        # 4. Обучение классификатора
        logger.info("\n[4/5] Обучение классификатора...")
        classifier = DeveloperLevelClassifier(
            n_estimators=100,
            class_weight='balanced',  # Учитываем дисбаланс классов
            random_state=42,
        )

        X_train, X_test, y_train, y_test = classifier.fit(
            it_df,
            levels,
            test_size=args.test_size,
        )

        logger.info(f"Размер обучающей выборки: {len(X_train)}")
        logger.info(f"Размер тестовой выборки: {len(X_test)}")

        # 5. Оценка модели
        logger.info("\n[5/5] Оценка модели...")
        report_str = classifier.evaluate(X_test, y_test, output_dir=output_dir)

        # Важность признаков
        feature_importance = classifier.get_feature_importance(top_n=20)
        importance_path = output_dir / "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False, encoding='utf-8')
        logger.info(f"Важность признаков сохранена: {importance_path}")

        logger.info("\n" + "=" * 70)
        logger.info("Классификация завершена успешно!")
        logger.info(f"Результаты сохранены в: {output_dir}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("Ошибка при выполнении классификации: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
