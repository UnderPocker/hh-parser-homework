"""
Базовый класс для обработчиков в паттерне Chain of Responsibility.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """
    Базовый класс обработчика в цепочке ответственности.
    
    Каждый обработчик может обработать запрос или передать его следующему
    обработчику в цепочке.
    """
    
    def __init__(self, next_handler: Optional['BaseHandler'] = None):
        """
        Инициализация обработчика.
        
        Args:
            next_handler: Следующий обработчик в цепочке.
        """
        self._next_handler = next_handler
    
    def set_next(self, handler: 'BaseHandler') -> 'BaseHandler':
        """
        Устанавливает следующий обработчик в цепочке.
        
        Args:
            handler: Следующий обработчик.
            
        Returns:
            Обработчик для цепочного вызова.
        """
        self._next_handler = handler
        return handler
    
    def handle(self, data: Any) -> Any:
        """
        Обрабатывает данные и передает их следующему обработчику.
        
        Args:
            data: Данные для обработки.
            
        Returns:
            Обработанные данные.
        """
        processed_data = self.process(data)
        
        if self._next_handler:
            return self._next_handler.handle(processed_data)
        
        return processed_data
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Выполняет конкретную обработку данных.
        
        Args:
            data: Данные для обработки.
            
        Returns:
            Обработанные данные.
        """
        pass
