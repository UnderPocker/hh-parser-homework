"""
Анализ качества модели и выводы о классификации.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Анализатор качества модели классификации."""

    @staticmethod
    def analyze_classification_report(
        report_str: str,
        y_test: pd.Series,
        y_pred: pd.Series,
    ) -> Dict[str, Any]:
        """
        Анализирует classification report и извлекает метрики.

        Args:
            report_str: Текст classification report.
            y_test: Реальные метки.
            y_pred: Предсказанные метки.

        Returns:
            Словарь с метриками.
        """
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=['junior', 'middle', 'senior'],
            output_dict=True,
        )
        
        return report_dict

    @staticmethod
    def generate_insights(
        report_dict: Dict[str, any],
        class_counts: pd.Series,
        feature_importance: pd.DataFrame,
    ) -> List[str]:
        """
        Генерирует выводы о качестве модели и возможных причинах ошибок.

        Args:
            report_dict: Словарь с метриками из classification report.
            class_counts: Количество примеров по классам.
            feature_importance: Важность признаков.

        Returns:
            Список выводов (insights).
        """
        insights = []
        
        # Анализ баланса классов
        total = class_counts.sum()
        proportions = class_counts / total
        
        insights.append("=" * 70)
        insights.append("АНАЛИЗ КАЧЕСТВА МОДЕЛИ И ВЫВОДЫ")
        insights.append("=" * 70)
        insights.append("")
        
        # 1. Баланс классов
        insights.append("1. БАЛАНС КЛАССОВ:")
        for level in ['junior', 'middle', 'senior']:
            count = class_counts.get(level, 0)
            prop = proportions.get(level, 0) * 100
            insights.append(f"   - {level.capitalize()}: {count} резюме ({prop:.1f}%)")
        
        max_prop = proportions.max()
        min_prop = proportions.min()
        imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')
        
        if imbalance_ratio > 3:
            insights.append(f"   Обнаружен значительный дисбаланс классов (соотношение {imbalance_ratio:.1f}:1)")
            insights.append("   → Это может снижать качество предсказания для минорных классов")
        elif imbalance_ratio > 2:
            insights.append(f"   Умеренный дисбаланс классов (соотношение {imbalance_ratio:.1f}:1)")
        else:
            insights.append("   ✓ Классы достаточно сбалансированы")
        
        insights.append("")
        
        # 2. Метрики по классам
        insights.append("2. МЕТРИКИ ПО КЛАССАМ:")
        for level in ['junior', 'middle', 'senior']:
            if level in report_dict:
                metrics = report_dict[level]
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1-score', 0)
                support = metrics.get('support', 0)
                
                insights.append(f"   {level.capitalize()}:")
                insights.append(f"     - Precision: {precision:.3f}")
                insights.append(f"     - Recall: {recall:.3f}")
                insights.append(f"     - F1-score: {f1:.3f}")
                insights.append(f"     - Support: {int(support)}")
                
                # Анализ качества
                if f1 < 0.5:
                    insights.append(f"     Низкое качество предсказания (F1 < 0.5)")
                elif f1 < 0.7:
                    insights.append(f"     Среднее качество предсказания (F1 < 0.7)")
                else:
                    insights.append(f"     Хорошее качество предсказания")
        
        insights.append("")
        
        # 3. Общие метрики
        macro_avg = report_dict.get('macro avg', {})
        weighted_avg = report_dict.get('weighted avg', {})
        
        insights.append("3. ОБЩИЕ МЕТРИКИ:")
        insights.append(f"   - Macro avg F1: {macro_avg.get('f1-score', 0):.3f}")
        insights.append(f"   - Weighted avg F1: {weighted_avg.get('f1-score', 0):.3f}")
        insights.append(f"   - Accuracy: {report_dict.get('accuracy', 0):.3f}")
        insights.append("")
        
        # 4. Важность признаков
        insights.append("4. ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
        top_features = feature_importance.head(10)
        for idx, row in top_features.iterrows():
            insights.append(f"   {idx + 1}. {row['feature']}: {row['importance']:.4f}")
        insights.append("")
        
        # 5. Возможные причины ошибок
        insights.append("5. ВОЗМОЖНЫЕ ПРИЧИНЫ ОШИБОК:")
        
        # Дисбаланс классов
        if imbalance_ratio > 3:
            insights.append("   - Значительный дисбаланс классов:")
            insights.append("     Модель лучше предсказывает доминирующий класс")
            insights.append("     Рекомендация: использовать техники балансировки (SMOTE, undersampling)")
        
        # Качество разметки
        if any(report_dict.get(level, {}).get('f1-score', 0) < 0.5 for level in ['junior', 'middle', 'senior']):
            insights.append("   - Низкое качество разметки или неоднозначность классов:")
            insights.append("     → Границы между junior/middle/senior могут быть размыты")
            insights.append("     → Некоторые резюме могут относиться к промежуточным уровням")
            insights.append("     → Рекомендация: улучшить правила определения уровня")
        
        # Недостаток признаков
        if len(feature_importance) < 10:
            insights.append("   - Недостаточно признаков для качественной классификации:")
            insights.append("     → Модель может не улавливать важные паттерны")
            insights.append("     → Рекомендация: добавить больше признаков (навыки, проекты, образование)")
        
        # Низкая важность признаков
        max_importance = feature_importance['importance'].max()
        if max_importance < 0.1:
            insights.append("   - Низкая важность признаков:")
            insights.append("     → Признаки слабо связаны с целевой переменной")
            insights.append("     → Рекомендация: провести feature engineering")
        
        insights.append("")
        
        # 6. Выводы
        insights.append("6. ВЫВОДЫ:")
        overall_f1 = weighted_avg.get('f1-score', 0)
        
        if overall_f1 >= 0.8:
            insights.append("   Модель показывает отличное качество (F1 >= 0.8)")
            insights.append("   PoC успешен: можно автоматически различать уровни разработчиков")
        elif overall_f1 >= 0.7:
            insights.append("   Модель показывает хорошее качество (F1 >= 0.7)")
            insights.append("   PoC успешен: модель пригодна для практического использования")
        elif overall_f1 >= 0.6:
            insights.append("   Модель показывает приемлемое качество (F1 >= 0.6)")
            insights.append("   PoC частично успешен: требуется доработка модели")
        else:
            insights.append("   Модель показывает низкое качество (F1 < 0.6)")
            insights.append("   PoC требует улучшения: необходимо пересмотреть подход")
        
        insights.append("")
        insights.append("=" * 70)
        
        return insights
