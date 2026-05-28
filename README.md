# ML Ops Homework 1

Docker-сервис для inference fraud-модели. Контейнер читает `test.csv` из
монтированной директории `input`, применяет препроцессинг и модель, затем
сохраняет результаты в `output`.

## Структура

```text
src/load_data.py        # загрузка входного test.csv
src/preprocess.py       # feature engineering
src/predict.py          # загрузка модели и скоринг
src/save_submission.py  # сохранение sample_submission.csv
src/main.py             # запуск всего inference pipeline
models/                 # компактная обученная модель
```

## Подготовка входных данных

Положите файл соревнования `test.csv` в директорию `input`:

```text
input/test.csv
```

Ожидаемые исходные колонки:

```text
transaction_time, merch, cat_id, amount, name_1, name_2, gender, street,
one_city, us_state, post_code, lat, lon, population_city, jobs,
merchant_lat, merchant_lon
```

## Локальный запуск

```bash
pip install -r requirements.txt
python -m src.main
```

После запуска в `output` появится Kaggle-файл:

```text
sample_submission.csv
```

## Docker

Сборка image:

```bash
docker build -t mlops-hw1 .
```

Запуск контейнера:

```bash
docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" mlops-hw1
```

На Windows PowerShell команда такая же:

```powershell
docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" mlops-hw1
```

## Модель

В проекте используется компактная hashed logistic regression модель,
обученная на `train.csv` из соревнования. Она сохранена в JSON и не требует
GPU. Скрипт `train_small_model.py` оставлен для воспроизведения обучения. Для
переобучения положите `train.csv` в `data/train.csv` или передайте путь явно:

```bash
python train_small_model.py --train-path data/train.csv
```
