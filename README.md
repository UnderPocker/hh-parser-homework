# Pipeline и регрессия для датасета HeadHunter

Проект: пайплайн обработки hh.csv (Chain of Responsibility) и регрессионная модель для предсказания зарплат. Веса модели хранятся в папке `resources/`.

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### 1. Пайплайн (hh.csv → x_data.npy, y_data.npy)

Через единое приложение или отдельный скрипт:

```bash
python app.py path/to/hh.csv
# или
python run_pipeline.py path/to/hh.csv
```

Рядом с CSV создаются `x_data.npy` и `y_data.npy`.

### 2. Обучение модели

На вход — выход пайплайна (директория с `x_data.npy` и `y_data.npy`). Веса сохраняются в `resources/`.

```bash
python train.py path/to/data_dir
# или
python train.py --x path/to/x_data.npy --y path/to/y_data.npy
```

Опции: `--alpha` — коэффициент L2-регуляризации Ridge (по умолчанию 1.0).

### 3. Предсказание зарплат (приложение)

На вход — путь к `x_data.npy`. В stdout выводится список зарплат в рублях (float), в формате JSON.

```bash
python app.py path/to/x_data.npy
```

Пример вывода (список float в рублях):

```json
[0.708, 0.714, 0.710, ...]
```

Логи пишутся в stderr, в stdout — только JSON-список.

**Единое приложение `app.py`:** по расширению файла выбирается режим: `.csv` — пайплайн, `.npy` — предсказание.

## Структура проекта

```
Homework-5/
├── app.py              CLI: python app path/to/x_data.npy → список зарплат (float)
├── run_pipeline.py     Пайплайн: python run_pipeline.py path/to/hh.csv
├── train.py            Обучение: python train.py data_dir → веса в resources/
├── resources/          Веса модели (model.npz) после train.py
├── model/              Модуль регрессии (Ridge)
├── pipeline/           Пайплайн Chain of Responsibility
├── requirements.txt
└── README.md
```

## Модель

- **Ridge-регрессия**.
- Признаки масштабируются.
- Сохранение/загрузка: `resources/model.npz`.

## Best practices и кодстайл

- Type hints, docstrings
- Логирование
- Обработка ошибок, валидация путей и данных
- PEP 8, модульная структура
