//! Сверка с Python: те же анкеты → те же вероятности классов, топ-10, блоки и сегменты.
//! Эталон создаётся командой `python tools/export_for_rust.py --golden 60`.

use phonepicker_server::{engine, model::Model};
use serde::Deserialize;

#[derive(Deserialize)]
struct Seg {
    title: String,
    top: Vec<i64>,
}

#[derive(Deserialize)]
struct Case {
    survey: engine::Survey,
    proba: Vec<f64>,
    top: Vec<i64>,
    blocks: Vec<Vec<i64>>,
    feasible: usize,
    best_cluster: usize,
    hint_key: Option<String>,
    segments: Vec<Seg>,
}

#[test]
fn matches_python() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let model = Model::from_json(&std::fs::read_to_string(format!("{dir}/model.json")).unwrap()).unwrap();
    let Ok(text) = std::fs::read_to_string(format!("{dir}/golden.json")) else {
        eprintln!("golden.json нет — пропускаю");
        return;
    };
    let cases: Vec<Case> = serde_json::from_str(&text).unwrap();
    let mut top_mismatch = 0;
    for (k, c) in cases.iter().enumerate() {
        let r = engine::recommend(&model, &c.survey, 10, true, true);
        for (a, b) in r.proba.iter().zip(&c.proba) {
            assert!((a - b).abs() < 1e-6, "анкета {k}: вероятности {:?} vs {:?}", r.proba, c.proba);
        }
        assert_eq!(r.feasible, c.feasible, "анкета {k}: число подходящих");
        assert_eq!(r.best_cluster, c.best_cluster, "анкета {k}: лучшая группа");
        assert_eq!(r.hint.as_ref().map(|h| h.key.clone()), c.hint_key, "анкета {k}: подсказка");
        let top: Vec<i64> = r.table.iter().map(|x| x.id).collect();
        if top != c.top {
            top_mismatch += 1;
            eprintln!("анкета {k}: топ Rust {top:?}\n           Python {:?}", c.top);
        }
        let blocks: Vec<Vec<i64>> = r.blocks.iter().map(|b| b.rows.iter().map(|x| x.id).collect()).collect();
        assert_eq!(blocks, c.blocks, "анкета {k}: блоки");
        let segs = engine::recommend_by_segments(&model, &c.survey, 10, true);
        assert_eq!(segs.len(), c.segments.len(), "анкета {k}: число сегментов");
        for (s, e) in segs.iter().zip(&c.segments) {
            assert_eq!(s.title, e.title);
            let ids: Vec<i64> = s.result.table.iter().map(|x| x.id).collect();
            assert_eq!(ids, e.top, "анкета {k}: сегмент {}", e.title);
        }
    }
    assert_eq!(top_mismatch, 0, "топ-10 отличается в {top_mismatch} анкетах из {}", cases.len());
    println!("совпало {} анкет из {}", cases.len(), cases.len());
}
