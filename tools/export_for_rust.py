"""Экспорт обученного конвейера в JSON для быстрого Rust-бэкенда (папка rust_backend/).

Python по-прежнему обучает модели (sklearn: метки, веса признаков, KMeans, MLP, справедливая цена).
Rust загружает только результат обучения и считает подбор для каждой анкеты:
полезность, жёсткие требования, прямой проход нейросети, мягкий бонус группы, разнообразие, блоки.

    python tools/export_for_rust.py                 # → rust_backend/model.json
    python tools/export_for_rust.py --golden 40     # + эталонные ответы Python для проверки Rust
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import db  # noqa: E402
from src.ml.classifier import PhonePipeline  # noqa: E402
from src.ml.features import FEATURE_NAMES_RU, PHONE_FEATURES  # noqa: E402
from src.ml.labeling import (CLASS_NAMES, IMPORTANCE_DEFAULTS, IMPORTANCE_FIELDS, IMPORTANCE_HINTS,  # noqa: E402
                             IMPORTANCE_TITLES, PRESET_HINTS, PRESETS, REQUIREMENT_TITLES, SCREEN_PREF,
                             SURVEY_FIELDS, random_survey)
from src.ml.price_model import PriceModel  # noqa: E402
from src.paths import ensure_runtime_files  # noqa: E402
from src.ml.recommender import full_name, recommend, recommend_by_segments  # noqa: E402

OUT = ROOT / "rust_backend" / "model.json"
GOLDEN = ROOT / "rust_backend" / "golden.json"

PHONE_COLS = ["id", "brand", "model", "os", "year", "price_rub", "ram_gb", "rom_gb", "battery_mah", "screen_in",
              "refresh_hz", "weight_g", "camera_mp", "antutu_k", "charge_w", "has_5g", "nfc", "esim", "ir_port",
              "active_h", "standby_d", "price_source"]


def _f(x) -> list:
    return np.asarray(x, dtype=float).round(10).tolist()


def export(pipe: PhonePipeline, fair: dict | None) -> dict:
    phones = pipe.phones_
    plist = []
    for _, p in phones.iterrows():
        row = {c: (p[c].item() if hasattr(p[c], "item") else p[c]) for c in PHONE_COLS}
        row["price_source"] = str(row.get("price_source") or "")
        row["full_name"] = full_name(p["brand"], p["model"])
        plist.append(row)
    sc, mlp = pipe.model_.named_steps["sc"], pipe.model_.named_steps["clf"]
    t = pipe.metrics_["test"]
    return {
        "version": 1,
        "trained_at": pipe.trained_at_,
        "features": PHONE_FEATURES,
        "feature_names_ru": [FEATURE_NAMES_RU[f] for f in PHONE_FEATURES],
        "survey_fields": SURVEY_FIELDS,
        "phones": plist,
        "z": _f(pipe.z_), "x": _f(pipe.x_),
        "labels": pipe.labels_.astype(int).tolist(),
        "clusters": pipe.clusters_.astype(int).tolist(),
        "centers": _f(pipe.centers_),
        "cluster_names": [pipe.cluster_names_[c] for c in range(pipe.k_)],
        "class_names": [CLASS_NAMES[c] for c in range(3)],
        "fair": {str(k): float(v) for k, v in (fair or {}).items()},
        "mlp": {
            "scaler_mean": _f(sc.mean_), "scaler_scale": _f(sc.scale_),
            "coefs": [_f(w) for w in mlp.coefs_], "intercepts": [_f(b) for b in mlp.intercepts_],
            "classes": mlp.classes_.astype(int).tolist(),
        },
        "metrics": {"accuracy": round(t["accuracy"], 4), "f1_macro": round(t["f1_macro"], 4),
                    "roc_auc": round(t["roc_auc_ovr"], 4), "surveys": pipe.n_surveys},
        "ui": {
            "importance": [{"key": k, "title": IMPORTANCE_TITLES[k], "hint": IMPORTANCE_HINTS[k],
                            "default": IMPORTANCE_DEFAULTS[k]} for k in IMPORTANCE_FIELDS],
            "presets": [{"name": n, "hint": PRESET_HINTS[n], "values": v} for n, v in PRESETS.items()],
            "screen_pref": [{"value": k, "title": v[0], "lo": v[1][0], "hi": v[1][1]} for k, v in SCREEN_PREF.items()],
            "requirement_titles": REQUIREMENT_TITLES,
        },
    }


def _ids(table) -> list[int]:
    return [int(i) for i in table["id"]] if len(table) else []


def golden(pipe, fair, n: int) -> list[dict]:
    """Эталонные анкеты и ответы Python — Rust-тест сверяет с ними свой результат."""
    rng = np.random.default_rng(123)
    cases = []
    for k in range(n):
        a = random_survey(rng)
        a["budget_min"] = float(rng.choice([0, 0, 0, 15000, 25000, 45000]))
        if a["budget_min"] > a["budget_rub"]:
            a["budget_min"] = 0.0
        t0 = time.perf_counter()
        res = recommend(pipe, a, top_n=10, fair=fair)
        dt = time.perf_counter() - t0
        segs = recommend_by_segments(pipe, a, top_n=10, fair=fair)
        cases.append({
            "survey": a, "python_ms": round(dt * 1000, 3),
            "proba": _f(res["proba"]), "top": _ids(res["table"]),
            "blocks": [[int(r["id"]) for r in b["rows"]] for b in res["blocks"]],
            "feasible": res["feasible"], "best_cluster": res["best_cluster"],
            "hint_key": (res["hint"] or {}).get("key"),
            "segments": [{"title": s["title"], "top": _ids(s["res"]["table"])} for s in segs],
        })
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=int, default=0, help="сколько эталонных анкет сохранить")
    args = ap.parse_args()
    ensure_runtime_files()
    db.init_db()
    phones = db.get_phones()
    pipe = PhonePipeline.load()
    if pipe is None or len(pipe.phones_) != len(phones):
        print("обучаю конвейер…")
        pipe = PhonePipeline().fit(phones, db.get_liked_feedback(), db.get_liked_feedback(False))
        pipe.save()
    pm = PriceModel.load(phones=pipe.phones_) or PriceModel().fit(pipe.phones_)
    fair = getattr(pm, "fair_", None)
    data = export(pipe, fair)
    # эталон скорости: тот же подбор на Python (sklearn + pandas) на этой машине
    rng = np.random.default_rng(7)
    times = []
    for _ in range(60):
        a = random_survey(rng)
        t0 = time.perf_counter()
        recommend(pipe, a, top_n=10, fair=fair)
        times.append((time.perf_counter() - t0) * 1000)
    data["python_bench_ms"] = round(float(np.median(times)), 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT}  ({OUT.stat().st_size / 1024:.0f} КБ, телефонов: {len(pipe.phones_)}, групп: {pipe.k_})")
    if args.golden:
        GOLDEN.write_text(json.dumps(golden(pipe, fair, args.golden), ensure_ascii=False), encoding="utf-8")
        print(f"{GOLDEN}  ({args.golden} анкет)")


if __name__ == "__main__":
    main()
