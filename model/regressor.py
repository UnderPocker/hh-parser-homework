"""
Регрессионная модель (Ridge) для предсказания зарплат в рублях.
Обучение и предсказание на numpy без sklearn для минимальных зависимостей.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Директория с весами относительно корня проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
DEFAULT_MODEL_PATH = RESOURCES_DIR / "model.npz"


class SalaryRegressor:
    """
    Ridge-регрессия для предсказания зарплаты по признакам.
    Признаки масштабируются (zero mean, unit variance) при обучении и предсказании.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        """
        Args:
            alpha: Коэффициент L2-регуляризации (Ridge).
            fit_intercept: Учитывать свободный член.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._fitted = False

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Масштабирование признаков по сохранённым mean/std."""
        if self._mean is None or self._std is None:
            return X
        std_safe = np.where(self._std == 0, 1.0, self._std)
        return (X - self._mean) / std_safe

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SalaryRegressor":
        """
        Обучает модель по (X, y).

        Args:
            X: Признаки, shape (n_samples, n_features).
            y: Целевая переменная (зарплата в рублях), shape (n_samples,).

        Returns:
            self.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n, d = X.shape
        if n == 0 or d == 0:
            raise ValueError("X не может быть пустым")

        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        std_safe = np.where(self._std == 0, 1.0, self._std)
        X_scaled = (X - self._mean) / std_safe

        if self.fit_intercept:
            X_design = np.hstack([np.ones((n, 1)), X_scaled])
        else:
            X_design = X_scaled

        # Ridge: w = (X'X + alpha*I)^(-1) X'y
        reg = self.alpha * np.eye(X_design.shape[1])
        reg[0, 0] = 0  # не регуляризуем intercept
        XtX = X_design.T @ X_design + reg
        Xty = X_design.T @ y
        try:
            w = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X_design, y, rcond=None)[0]

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        self._fitted = True
        logger.info(
            "Модель обучена: coef shape %s, intercept %s",
            self.coef_.shape,
            self.intercept_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание зарплат в рублях.

        Args:
            X: Признаки, shape (n_samples, n_features).

        Returns:
            Предсказанные зарплаты, shape (n_samples,).
        """
        if not self._fitted or self.coef_ is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit() или load().")

        X = np.asarray(X, dtype=np.float64)
        X_scaled = self._scale(X)
        return (X_scaled @ self.coef_) + self.intercept_

    def save(self, path: Optional[Path] = None) -> None:
        """
        Сохраняет веса модели в resources.

        Args:
            path: Путь к файлу. По умолчанию resources/model.npz.
        """
        path = path or DEFAULT_MODEL_PATH
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            coef=self.coef_,
            intercept=np.array(self.intercept_),
            mean=self._mean,
            std=self._std,
        )
        logger.info("Модель сохранена: %s", path)

    def load(self, path: Optional[Path] = None) -> "SalaryRegressor":
        """
        Загружает веса модели из resources.

        Args:
            path: Путь к файлу. По умолчанию resources/model.npz.

        Returns:
            self.
        """
        path = path or DEFAULT_MODEL_PATH
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {path}. Сначала запустите train.py."
            )

        data = np.load(path, allow_pickle=False)
        self.coef_ = np.asarray(data["coef"])
        intr = data["intercept"]
        self.intercept_ = float(np.asarray(intr).ravel().item())
        self._mean = data["mean"] if "mean" in data else None
        self._std = data["std"] if "std" in data else None
        self._fitted = True
        logger.info("Модель загружена: %s", path)
        return self
