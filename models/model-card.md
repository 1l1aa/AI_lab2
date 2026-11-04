# Model Card — FishGrow Weight Predictor

## Назначение
Предсказание массы рыбы (г) по геометрическим измерениям и виду.
Задача: регрессия. Целевая переменная: `Weight` (граммы).

## Входные признаки
- `Volume = Length3 · Height · Width` (см³) — производный геометрический признак
- `Length3` (см) — полная длина
- `Height` (см) — высота
- `Width` (см) — ширина
- `Species` — категориальный, one-hot encoding (7 видов,
  `handle_unknown="ignore"` для новых категорий)

## Исключённые признаки
- `Length1`, `Length2` — мультиколлинеарность (r > 0.98 с `Length3`),
  дублируют информацию и увеличивают дисперсию коэффициентов линейной модели.

## Выход
`Weight` — масса рыбы, граммы (после обратного `expm1` от прогноза модели).

## Данные
- Источник: `assets/data/Fish.csv` (159 × 7).
- После исключения ошибки `Weight = 0`: **158 строк**.
- Разбиение 60/20/20 (`GroupShuffleSplit` по округлённым габаритам,
  seed=42): **train=94, val=32, test=32**.
- Версия данных: `fish.csv@v1`.

## Препроцессинг
Весь препроцессинг настроен **внутри Pipeline** (fit только на train,
нет утечки данных на CV):
- Числовые признаки: `SimpleImputer(strategy="median")` → `StandardScaler`
- Категориальные: `SimpleImputer(strategy="most_frequent")` →
  `OneHotEncoder(handle_unknown="ignore")`
- Целевая переменная: `TransformedTargetRegressor(func=log1p, inverse_func=expm1)`

## Лучшая модель
- **Эксперимент:** E1 (`linear_regression_log_target`)
- **Модель:** LinearRegression с `TransformedTargetRegressor(log1p/expm1)`

- **Обоснование выбора:** логарифмирование `Weight` сжимает длинный правый
хвост распределения (Pike до 1650 г), что даёт MSE в 3.3 раза ниже, чем
у Ridge без `log-target` (E2) и RandomForest (E3).

## Метрики (тестовая выборка, 32 объекта)

| Метрика | Значение |
|---------|----------|
| Test MSE | **1593.63 г²** |
| Test RMSE | **39.92 г** |
| Test MAE | **24.98 г** |
| Test R² | **0.9844** |

**Bootstrap 95% CI (N=1000, seed=42):**
- RMSE: **[21.45; 59.33] г**
- R²: **[0.9573; 0.9952]**

Точечные оценки попадают в середину интервалов — модель стабильна.
R² остаётся положительным и > 0.95 на всех 1000 ресемплах, то есть модель
**никогда** не хуже константного прогноза (baseline E0: R² = −0.0166).

## Сравнение с другими экспериментами

| ID | Модель | Test MSE | Test R² |
|----|--------|----------|---------|
| E0 | DummyRegressor (среднее) | 103849.18 | −0.0166 |
| **E1** | **LinearRegression + log1p** | **1593.63** | **0.9844** |
| E2 | Ridge + GridSearchCV(α) | 5324.41 | 0.9479 |
| E3 | RandomForest | 4148.50 | 0.9594 |
| E4 | LinearRegression без Species | 6426.56 | 0.9371 |

## Ограничения

1. **7 видов в обучении.** Новый вид → `OneHotEncoder(handle_unknown="ignore")`
 вернёт вектор из нулей → прогноз ненадёжен (будет опираться только
 на геометрию без учёта вида).
2. **Дисбаланс видов** (в train: Perch=39, Bream=24, Roach=14, Pike=11,
 Smelt=10, Parkki=8, Whitefish=4). Для редких видов (Whitefish, Smelt)
 метрики менее надёжны.
3. **Крупные особи** (Pike > 1500 г) в тесте отсутствуют; в train их
 2–3 экземпляра. На таких рыбах возможна недооценка массы.
4. **Аномалии в данных.** Bream с `Volume=2632`, `Weight=340` — модель
 предсказывает ~513 г по геометрии. Это ограничение данных, а не модели
 (топ-1 ошибка на тесте: 172.74 г).
5. **Одна популяция.** Данные собраны в одном регионе (см. ЛР№1);
 применимость модели на других популяциях ограничена.
6. **Физические ограничения соблюдены** — благодаря `log1p/expm1`
 прогноз всегда > 0 (на тесте отрицательных прогнозов: 0).

## Анализ ошибок (кратко)
- **Наибольшая MAE — Bream (46.46 г, n=10)**, наименьшая — Smelt (2.26 г, n=1).
- **MAE по диапазонам массы:** <200 г — 7.05, 200–500 г — 41.74,
500–1000 г — 33.97. Крупные ошибки сосредоточены в среднем диапазоне
из-за разброса массы Bream при схожих габаритах.
- **Топ-1 ошибка** — аномальный Bream (340 → 513 г), источник — данные.
- Подробный анализ: `reports/lab02-report.md`, раздел 3;
CSV: `reports/error_*.csv`.

## Артефакты
- Модель: `models/best_model.joblib`
- Метрики: `models/best_model_metrics.json`, `reports/test_metrics.json`
- Отчёт: `reports/lab02-report.md`
- Графики: `reports/figures/E1_pred_vs_fact.png`, `reports/figures/E1_residuals.png`,
`reports/figures/06_bootstrap_ci.png`
- MLflow: `mlruns/` (5 run'ов), снимок — `reports/mlruns_snapshot.json`

## Воспроизведение
```bash
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.bootstrap_ci
pytest tests/ -v
```
## Версия
**v0.1** — ЛР№2, ветка `lab/02-regression`.