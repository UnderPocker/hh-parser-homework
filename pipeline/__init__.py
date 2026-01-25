"""
Pipeline модуль для обработки данных с использованием паттерна Chain of Responsibility.
"""

from .pipeline import DataPipeline
from .base_handler import BaseHandler
from .handlers import (
    LoadDataHandler,
    CleanDataHandler,
    EncodeCategoricalHandler,
    HandleMissingValuesHandler,
    FeatureEngineeringHandler,
    SplitDataHandler,
    SaveDataHandler,
)

__all__ = [
    'DataPipeline',
    'BaseHandler',
    'LoadDataHandler',
    'CleanDataHandler',
    'EncodeCategoricalHandler',
    'HandleMissingValuesHandler',
    'FeatureEngineeringHandler',
    'SplitDataHandler',
    'SaveDataHandler',
]
