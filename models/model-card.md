# Model Card — FishGrow Weight Predictor

## Назначение
Предсказание массы рыбы (г) по геометрическим измерениям.

## Входные признаки
- `Volume = Length3 · Height · Width` (см³)
- `Length3`, `Height`, `Width` (см)
- `Species` (one-hot encoding, 7 видов)

## Исключённые признаки
- `Length1`, `Length2` — мультиколлинеарность (r > 0.98 с `Length3`).

## Выход
`Weight` — масса рыбы, граммы.

## Данные
- Источник: `assets/data/Fish.csv` (159 × 7).
- После исключения ошибки `Weight = 0`: 158 строк.
- Разбиение 60/20/20: train=94, val=32, test=32.
- Seed: 42.
- Крупные особи Pike (1550–1650 г) оставлены намеренно.

## Лучшая модель
- **Эксперимент:** E3 (`linear_all_features_with_species`)
- **Модель:** LinearRegression
- **Пайплайн:** StandardScaler + OneHotEncoder + LinearRegression.

## Метрики
| Набор | RMSE, г | MAE, г | R² |
|---|---|---|---|
| Validation | 54.50 | 42.42 | 0.9779 |
| Test | 72.88 | 53.49 | 0.9480 |

## Ограничения
1. Обучена на 7 видах. Новый вид → one-hot вектор из нулей → ненадёжное предсказание.
2. Дисбаланс видов (Whitefish=6, Smelt=14).
3. Крупные рыбы (Pike >1500 г) могут недооцениваться.
4. Данные собраны в одной популяции (см. t-тест в ЛР№1), применимость на других ограничена.

## Воспроизведение
```bash
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml

Версия
v0.1, 2026-11-04
```