import pandas as pd
import numpy as np

def check_shape_and_dtypes(df):
    return {"shape": df.shape, "columns": list(df.columns), "dtypes": df.dtypes.astype(str).to_dict()}

def check_duplicates(df):
    full_dups = df.duplicated().sum()
    numeric_cols = df.select_dtypes(include=np.number).columns
    near_dups = df.duplicated(subset=numeric_cols, keep=False).sum()
    return {"full_duplicates": int(full_dups), "near_duplicates_rounded": int(near_dups)}

def check_missing_and_inf(df):
    missing = df.isna().sum().to_dict()
    inf_plus = np.isinf(df.select_dtypes(include=np.number)).sum().to_dict()
    return {"missing": missing, "inf_plus": inf_plus}

def check_non_positive(df):
    size_cols = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
    issues = {}
    for col in ['Weight'] + size_cols:
        count = int((df[col] <= 0).sum())
        rows = df[df[col] <= 0].index.tolist()
        issues[col] = {"count": count, "rows": rows}
    return issues

def check_length_consistency(df):
    mask = (df['Length3'] < df['Length2']) | (df['Length2'] < df['Length1'])
    return df[mask][['Species', 'Length1', 'Length2', 'Length3']]

def check_impossible_combos(df):
    # Проверка: ширина не может быть больше высоты
    mask = df['Width'] > df['Height']
    return df[mask][['Species', 'Width', 'Height']]

def check_category_balance(df):
    counts = df['Species'].value_counts()
    rare = counts[counts < 10]
    return {"counts": counts.to_dict(), "rare_categories": rare.index.tolist()}

def check_category_spelling(df):
    # Проверяем на лишние пробелы в начале/конце
    bad_spelling = df[df['Species'] != df['Species'].str.strip()]
    return bad_spelling.index.tolist()

def check_near_duplicates(df):
    # Округляем числа и ищем почти одинаковые строки
    df_rounded = df.select_dtypes(include=np.number).round(0)
    mask = df_rounded.duplicated(keep=False)
    return df[mask]

def check_unique_fields(df):
    unique_fields = [col for col in df.columns if df[col].nunique() == len(df)]
    return unique_fields

def check_leak_candidates(df, target_col='Weight'):
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()[target_col].abs()
    leaks = corr[corr > 0.98].index.tolist()
    if target_col in leaks:
        leaks.remove(target_col)
    return leaks

def run_all_checks(df):
    results = {}
    results["shape"] = check_shape_and_dtypes(df)["shape"]
    results["columns"] = check_shape_and_dtypes(df)["columns"]
    results["quality"] = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "miss_%": df.isna().mean() * 100,
        "unique": df.nunique()
    })
    results["duplicates"] = check_duplicates(df)
    results["missing_inf"] = check_missing_and_inf(df)
    results["non_positive"] = check_non_positive(df)
    results["length_issue"] = check_length_consistency(df)
    results["impossible"] = {"Width_gt_Height": check_impossible_combos(df)}
    results["categories"] = {"Species": check_category_balance(df)}
    results["spelling_issue"] = {"Species": check_category_spelling(df)}
    results["near_doubles"] = check_near_duplicates(df)
    results["unique_fields"] = check_unique_fields(df)
    results["leak_candidates"] = check_leak_candidates(df)
    return results

def summary(results):
    rows = []
    rows.append({"Проверка": "Форма таблицы", "Значение": f"{results['shape'][0]} x {results['shape'][1]}", "Статус": "OK"})
    rows.append({"Проверка": "Полные дубликаты", "Значение": results['duplicates']['full_duplicates'], "Статус": "OK" if results['duplicates']['full_duplicates'] == 0 else "⚠️"})
    rows.append({"Проверка": "Почти-дубликаты", "Значение": results['duplicates']['near_duplicates_rounded'], "Статус": "⚠️" if results['duplicates']['near_duplicates_rounded'] > 0 else "OK"})
    non_pos_count = sum(v['count'] for v in results['non_positive'].values())
    rows.append({"Проверка": "Неположительные значения", "Значение": non_pos_count, "Статус": "⚠️" if non_pos_count > 0 else "OK"})
    rows.append({"Проверка": "Невозможные комбинации", "Значение": len(results['impossible']['Width_gt_Height']), "Статус": "⚠️" if len(results['impossible']['Width_gt_Height']) > 0 else "OK"})
    rows.append({"Проверка": "Согласованность длин", "Значение": len(results['length_issue']), "Статус": "⚠️" if len(results['length_issue']) > 0 else "OK"})
    rows.append({"Проверка": "Уникальные поля", "Значение": len(results['unique_fields']), "Статус": "OK" if len(results['unique_fields']) == 0 else "⚠️"})
    rows.append({"Проверка": "Кандидаты на утечку", "Значение": len(results['leak_candidates']), "Статус": "OK" if len(results['leak_candidates']) == 0 else "⚠️"})
    return pd.DataFrame(rows)