"""
Визуализация баланса классов для классификации уровней разработчиков.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ClassBalanceVisualizer:
    """Визуализатор баланса классов для классификации."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Инициализация визуализатора.

        Args:
            output_dir: Директория для сохранения графиков.
        """
        self.output_dir = output_dir or Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_class_balance(
        self,
        y: pd.Series,
        title: str = "Распределение уровней разработчиков",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Строит графики баланса классов.

        Args:
            y: Series с метками классов (junior/middle/senior).
            title: Заголовок графика.
            save_path: Путь для сохранения (по умолчанию output_dir/class_balance.png).
        """
        logger.info("Построение графиков баланса классов...")

        if save_path is None:
            save_path = self.output_dir / "class_balance.png"

        # Подсчет классов
        class_counts = y.value_counts().sort_index()
        class_proportions = y.value_counts(normalize=True).sort_index() * 100

        # Создаем фигуру с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # График 1: Столбчатая диаграмма количества
        colors = ['#3498db', '#2ecc71', '#e74c3c']  # Синий, зеленый, красный
        bars = ax1.bar(
            class_counts.index,
            class_counts.values,
            color=colors[:len(class_counts)],
            alpha=0.7,
            edgecolor='black',
            linewidth=1.5,
        )
        ax1.set_xlabel('Уровень разработчика', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Количество резюме', fontsize=12, fontweight='bold')
        ax1.set_title('Количество резюме по уровням', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
            )

        # График 2: Круговая диаграмма пропорций
        wedges, texts, autotexts = ax2.pie(
            class_proportions.values,
            labels=class_proportions.index,
            autopct='%1.1f%%',
            colors=colors[:len(class_proportions)],
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
        )
        ax2.set_title('Процентное распределение', fontsize=14, fontweight='bold')

        # Общий заголовок
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"График сохранен: {save_path}")
        plt.close()

    def plot_class_distribution_detailed(
        self,
        y: pd.Series,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Строит детальный график распределения классов с дополнительной информацией.

        Args:
            y: Series с метками классов.
            save_path: Путь для сохранения.
        """
        if save_path is None:
            save_path = self.output_dir / "class_distribution_detailed.png"

        fig, ax = plt.subplots(figsize=(10, 6))

        class_counts = y.value_counts().sort_index()
        colors_map = {'junior': '#3498db', 'middle': '#2ecc71', 'senior': '#e74c3c'}
        colors = [colors_map.get(cls, '#95a5a6') for cls in class_counts.index]

        bars = ax.barh(
            class_counts.index,
            class_counts.values,
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5,
        )

        ax.set_xlabel('Количество резюме', fontsize=12, fontweight='bold')
        ax.set_ylabel('Уровень разработчика', fontsize=12, fontweight='bold')
        ax.set_title('Распределение уровней разработчиков (детальный вид)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Добавляем значения
        for i, (idx, val) in enumerate(class_counts.items()):
            ax.text(
                val,
                i,
                f'  {int(val)} ({val/len(y)*100:.1f}%)',
                va='center',
                fontsize=11,
                fontweight='bold',
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Детальный график сохранен: {save_path}")
        plt.close()
