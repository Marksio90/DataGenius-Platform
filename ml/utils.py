
from __future__ import annotations
import pandas as pd

def auto_pick_target(df: pd.DataFrame) -> str | None:
    for cand in ["AveragePrice","price","target","y","label"]:
        if cand in df.columns: return cand
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.lower().startswith("unnamed")]
    return num[-1] if num else None

def recommendations_text(target: str, problem: str, top_features: list[str]) -> str:
    tips = []
    if problem == "regression":
        tips.append("Zmniejsz rozrzut danych (np. transformacje, outliers) — poprawi RMSE/MAE.")
        tips.append("Dodaj cechy kalendarzowe (rok/miesiąc/dzień/tydzień), jeśli masz datę — model lepiej uchwyci sezonowość.")
        tips.append("Sprawdź segmenty kategorii z najwyższym błędem — tam jest najwięcej do ugrania.")
    else:
        tips.append("Zbalansuj klasy (class_weight/oversampling), jeśli są nierówne — F1/ROC-AUC wzrośnie.")
        tips.append("Dopasuj próg decyzyjny pod cel biznesowy (minimalizuj FP lub FN).")
        tips.append("Zbadaj cechy o najwyższej ważności: czy są stabilne i sensowne biznesowo?")
    header = ""
    if top_features:
        header = f"Najważniejsze cechy dla `{target}`: {', '.join(top_features[:5])}. Skup się na ich jakości i stabilności.\n"
    return (header + "- " + "\n- ".join(tips)).strip()

