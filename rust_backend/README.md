# Быстрый подбор смартфона на Rust

Python по-прежнему **обучает** модели: метки классов, веса признаков, KMeans по шпилям, нейросеть MLP и справедливую цену.
Rust **считает подбор** по анкете: полезность телефонов, жёсткие требования, прямой проход нейросети, мягкий бонус группы,
разнообразие выдачи, блоки и ценовые сегменты. Логика — построчный порт `src/ml/recommender.py` и `src/ml/labeling.py`.

```
Python (обучение, раз в N дней)            Rust (каждый запрос)
phones.db → PhonePipeline.fit()  ──► model.json ──► phonepicker-server ──► HTTP API ──► index.html
            PriceModel.fit()          (≈380 КБ)       (один файл ≈2 МБ)
```

## Запуск

```bash
# 1. Выгрузить обученную модель (из корня проекта)
python tools/export_for_rust.py --golden 60     # → rust_backend/model.json и golden.json

# 2. Собрать и запустить сервер
cd rust_backend
cargo build --release
./target/release/phonepicker-server              # Windows: target\release\phonepicker-server.exe
# открыть http://localhost:8080
```

Модель и страница встроены в бинарник при сборке, поэтому для запуска на другой машине нужен только
сам файл `phonepicker-server`. Python там не нужен. Новую модель без пересборки можно подложить так:
`MODEL_PATH=model.json ./phonepicker-server`. Порт меняется переменной `PORT`.

Страницу `static/index.html` можно открыть и отдельно, файлом или с другого хоста: она обратится к `http://localhost:8080`
либо к адресу из параметра `?api=http://сервер:8080`. CORS на сервере открыт.

## API

| Метод | Путь | Что делает |
|---|---|---|
| `POST` | `/api/recommend` | Анкета (JSON, поля как в `SURVEY_FIELDS` + `budget_min`, `top_n`) → класс, вероятности, топ-N, блоки, сегменты, `compute_us` |
| `GET` | `/api/meta` | Ползунки, пресеты, группы, метрики модели — для построения формы |
| `GET` | `/api/bench?n=5000` | Замер скорости на n случайных анкетах |

Пример запроса:

```bash
curl -X POST localhost:8080/api/recommend -H 'content-type: application/json' \
     -d '{"budget_rub": 40000, "imp_camera": 9, "photo": 1, "need_nfc": 1, "min_rom": 256}'
```

Поля, которых нет в запросе, берутся по умолчанию, как в анкете приложения.

## Проверка, что Rust считает так же, как Python

```bash
cargo test --release -- --nocapture     # совпало 60 анкет из 60
```

Тест берёт 60 случайных анкет из `golden.json` (их ответы посчитал Python) и сверяет вероятности классов
(до 1e-6), число подходящих моделей, лучшую группу, подсказку, топ-10, блоки выдачи и сегменты.

## Скорость

На одной и той же машине и одном ядре:

| | Один подбор |
|---|---|
| Python: sklearn + pandas, `recommend()` | ≈ 8,8 мс |
| Rust: `engine::recommend()` | ≈ 50–60 мкс |

Это примерно в 150 раз быстрее, или около 15 000 подборов в секунду на ядро. Замер запускается кнопкой на странице или запросом `/api/bench`.

## Файлы

- `src/model.rs` — загрузка `model.json`, прямой проход MLP, предрасчёт мягкой принадлежности группам.
- `src/engine.rs` — подбор: полезность, требования, оценка, разнообразие, блоки, сегменты, подсказки.
- `src/main.rs` — HTTP-сервер (axum).
- `static/index.html` — веб-интерфейс. Шрифты Unbounded и Onest (SIL OFL) лежат в `static/fonts` и тоже встроены в бинарник.
- `tests/golden.rs` — сверка с Python.
