"""
Класс для управления пайплайном обработки данных.
"""

import logging
from typing import Optional
from .base_handler import BaseHandler
from .handlers import (
    LoadDataHandler,
    CleanDataHandler,
    HandleMissingValuesHandler,
    EncodeCategoricalHandler,
    FeatureEngineeringHandler,
    SplitDataHandler,
    SaveDataHandler,
)

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Класс для управления пайплайном обработки данных
    с использованием паттерна Chain of Responsibility.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Инициализация пайплайна.
        
        Args:
            output_dir: Директория для сохранения результатов.
        """
        self.output_dir = output_dir
        self._build_pipeline()
    
    def _build_pipeline(self) -> None:
        """Строит цепочку обработчиков."""
        # Создаем обработчики
        load_handler = LoadDataHandler()
        clean_handler = CleanDataHandler()
        missing_handler = HandleMissingValuesHandler(strategy='drop')
        encode_handler = EncodeCategoricalHandler()
        feature_handler = FeatureEngineeringHandler()
        split_handler = SplitDataHandler()
        save_handler = SaveDataHandler(output_dir=self.output_dir)
        
        # Строим цепочку
        load_handler.set_next(clean_handler) \
                   .set_next(missing_handler) \
                   .set_next(encode_handler) \
                   .set_next(feature_handler) \
                   .set_next(split_handler) \
                   .set_next(save_handler)
        
        self.pipeline = load_handler
    
    def process(self, input_path: str) -> None:
        """
        Запускает обработку данных через пайплайн.
        
        Args:
            input_path: Путь к входному CSV файлу.
        """
        logger.info(f"Запуск пайплайна обработки данных из {input_path}")
        try:
            self.pipeline.handle(input_path)
            logger.info("Пайплайн успешно завершен")
        except Exception as e:
            logger.error(f"Ошибка при выполнении пайплайна: {e}", exc_info=True)
            raise
