"""
Конкретные обработчики для пайплайна обработки данных.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from .base_handler import BaseHandler

logger = logging.getLogger(__name__)


class LoadDataHandler(BaseHandler):
    """Обработчик для загрузки данных из CSV файла."""
    
    def process(self, data: Any) -> pd.DataFrame:
        """
        Загружает данные из CSV файла.
        
        Args:
            data: Путь к CSV файлу (строка).
            
        Returns:
            DataFrame с загруженными данными.
        """
        if isinstance(data, str):
            logger.info(f"Загрузка данных из {data}")
            df = pd.read_csv(data, encoding='utf-8')
            logger.info(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
            return df
        return data


class CleanDataHandler(BaseHandler):
    """Обработчик для очистки данных."""
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Очищает данные: удаляет дубликаты, лишние колонки.
        
        Args:
            data: DataFrame для очистки.
            
        Returns:
            Очищенный DataFrame.
        """
        logger.info("Очистка данных...")
        df = data.copy()
        
        # Удаление колонки с индексом, если она есть
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            logger.info("Удалена колонка 'Unnamed: 0'")
        
        # Удаление дубликатов
        initial_len = len(df)
        df = df.drop_duplicates()
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Удалено {removed} дубликатов")
        
        logger.info(f"После очистки: {len(df)} строк, {len(df.columns)} колонок")
        return df


class HandleMissingValuesHandler(BaseHandler):
    """Обработчик для обработки пропущенных значений."""
    
    def __init__(self, next_handler: Optional[BaseHandler] = None, 
                 strategy: str = 'drop'):
        """
        Инициализация обработчика пропущенных значений.
        
        Args:
            next_handler: Следующий обработчик.
            strategy: Стратегия обработки ('drop', 'mean', 'median', 'mode').
        """
        super().__init__(next_handler)
        self.strategy = strategy
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает пропущенные значения.
        
        Args:
            data: DataFrame с пропущенными значениями.
            
        Returns:
            DataFrame без пропущенных значений.
        """
        logger.info("Обработка пропущенных значений...")
        df = data.copy()
        
        # Определяем целевую колонку (последняя колонка)
        target_col = df.columns[-1]
        
        # Удаляем строки, где целевая переменная отсутствует
        initial_len = len(df)
        df = df.dropna(subset=[target_col])
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Удалено {removed} строк с пропущенной целевой переменной")
        
        # Обработка пропущенных значений в признаках
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        categorical_cols = categorical_cols.drop(target_col, errors='ignore')
        
        # Для числовых колонок
        if len(numeric_cols) > 0:
            if self.strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif self.strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            else:
                df = df.dropna(subset=numeric_cols)
        
        # Для категориальных колонок
        if len(categorical_cols) > 0:
            if self.strategy == 'mode':
                for col in categorical_cols:
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col] = df[col].fillna(mode_value[0])
            else:
                df = df.dropna(subset=categorical_cols)
        
        logger.info(f"После обработки пропущенных значений: {len(df)} строк")
        return df


class EncodeCategoricalHandler(BaseHandler):
    """Обработчик для кодирования категориальных переменных."""
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Кодирует категориальные переменные.
        
        Args:
            data: DataFrame с категориальными переменными.
            
        Returns:
            DataFrame с закодированными переменными.
        """
        logger.info("Кодирование категориальных переменных...")
        df = data.copy()
        
        # Определяем целевую колонку
        target_col = df.columns[-1]
        
        # Получаем категориальные колонки (исключая целевую)
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        categorical_cols = categorical_cols.drop(target_col, errors='ignore')
        
        # Label encoding для категориальных переменных
        # Используем pandas factorize для кодирования
        label_encoders = {}
        for col in categorical_cols:
            df[col], unique_values = pd.factorize(df[col].astype(str))
            label_encoders[col] = unique_values
            logger.info(f"Закодирована колонка: {col} ({len(unique_values)} уникальных значений)")
        
        logger.info(f"Закодировано {len(categorical_cols)} категориальных колонок")
        return df


class FeatureEngineeringHandler(BaseHandler):
    """Обработчик для создания новых признаков."""
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет feature engineering.
        
        Args:
            data: DataFrame для обработки.
            
        Returns:
            DataFrame с новыми признаками.
        """
        logger.info("Feature engineering...")
        df = data.copy()
        
        # Здесь можно добавить создание новых признаков
        # Например, нормализация, создание взаимодействий и т.д.
        
        logger.info("Feature engineering завершен")
        return df


class SplitDataHandler(BaseHandler):
    """Обработчик для разделения данных на X и y."""
    
    def process(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Разделяет данные на признаки (X) и целевую переменную (y).
        
        Args:
            data: DataFrame с данными.
            
        Returns:
            Кортеж (X, y) в виде numpy массивов.
        """
        logger.info("Разделение данных на X и y...")
        df = data.copy()
        
        # Последняя колонка - целевая переменная
        target_col = df.columns[-1]
        y = df[target_col]
        
        # Если целевая переменная не числовая, кодируем её
        if not pd.api.types.is_numeric_dtype(y):
            logger.info(f"Кодирование целевой переменной: {target_col}")
            y, _ = pd.factorize(y.astype(str))
            y = pd.Series(y)
        
        y = y.values
        X = df.drop(columns=[target_col]).values
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        return (X, y)


class SaveDataHandler(BaseHandler):
    """Обработчик для сохранения данных в .npy файлы."""
    
    def __init__(self, next_handler: Optional[BaseHandler] = None,
                 output_dir: Optional[str] = None):
        """
        Инициализация обработчика сохранения.
        
        Args:
            next_handler: Следующий обработчик (не используется).
            output_dir: Директория для сохранения файлов.
        """
        super().__init__(next_handler)
        self.output_dir = output_dir or ""
    
    def process(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сохраняет данные в .npy файлы.
        
        Args:
            data: Кортеж (X, y) с данными.
            
        Returns:
            Исходные данные без изменений.
        """
        X, y = data
        
        # Убеждаемся, что данные в числовом формате
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        import os
        output_path = self.output_dir if self.output_dir else os.path.dirname(os.path.abspath('.'))
        
        x_path = os.path.join(output_path, 'x_data.npy')
        y_path = os.path.join(output_path, 'y_data.npy')
        
        logger.info(f"Сохранение данных в {x_path} и {y_path}")
        np.save(x_path, X)
        np.save(y_path, y)
        
        logger.info(f"Данные успешно сохранены. X: {X.shape}, y: {y.shape}")
        return data
