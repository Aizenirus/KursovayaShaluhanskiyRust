//! Загрузка модели, обученной в Python (tools/export_for_rust.py → model.json).
//!
//! Обозначения:
//!   n  — число телефонов, nf — число признаков телефона (18), k — число групп (кластеров);
//!   z  — стандартизованные признаки телефонов (n × nf), как `pipe.z_` в Python;
//!   x  — взвешенное пространство кластеризации (n × nf), как `pipe.x_`;
//!   centers — центры групп в пространстве x (k × nf);
//!   p  — мягкая принадлежность телефона группам (n × k): softmax по расстояниям до центров.
//!        От анкеты не зависит, поэтому считается один раз при загрузке.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Phone {
    pub id: i64,
    pub brand: String,
    pub model: String,
    pub full_name: String,
    pub os: String,
    pub year: f64,
    pub price_rub: f64,
    pub ram_gb: f64,
    pub rom_gb: f64,
    pub battery_mah: f64,
    pub screen_in: f64,
    pub refresh_hz: f64,
    pub weight_g: f64,
    pub camera_mp: f64,
    pub antutu_k: f64,
    pub charge_w: f64,
    pub has_5g: f64,
    pub nfc: f64,
    pub esim: f64,
    pub ir_port: f64,
    pub active_h: f64,
    pub standby_d: f64,
    #[serde(default)]
    pub price_source: String,
}

#[derive(Deserialize)]
struct RawMlp {
    scaler_mean: Vec<f64>,
    scaler_scale: Vec<f64>,
    coefs: Vec<Vec<Vec<f64>>>,
    intercepts: Vec<Vec<f64>>,
    classes: Vec<usize>,
}

#[derive(Deserialize)]
struct RawModel {
    trained_at: String,
    features: Vec<String>,
    feature_names_ru: Vec<String>,
    survey_fields: Vec<String>,
    phones: Vec<Phone>,
    z: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
    labels: Vec<usize>,
    clusters: Vec<usize>,
    centers: Vec<Vec<f64>>,
    cluster_names: Vec<String>,
    class_names: Vec<String>,
    fair: HashMap<String, f64>,
    mlp: RawMlp,
    metrics: serde_json::Value,
    ui: serde_json::Value,
    #[serde(default)]
    python_bench_ms: Option<f64>,
}

/// Один полносвязный слой: y = W·x + b, W хранится построчно (in × out), как `coefs_` в sklearn.
pub struct Layer {
    pub w: Vec<f64>,
    pub b: Vec<f64>,
    pub n_in: usize,
    pub n_out: usize,
}

/// MLPClassifier из sklearn: StandardScaler → ReLU-слои → softmax.
pub struct Mlp {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub layers: Vec<Layer>,
    pub classes: Vec<usize>,
}

impl Mlp {
    /// Вероятности классов (индекс = номер класса 0/1/2).
    pub fn predict_proba(&self, input: &[f64]) -> Vec<f64> {
        let mut h: Vec<f64> = input
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((v, m), s)| (v - m) / s)
            .collect();
        let last = self.layers.len() - 1;
        for (li, layer) in self.layers.iter().enumerate() {
            let mut out = layer.b.clone();
            for (i, &hi) in h.iter().enumerate() {
                if hi == 0.0 {
                    continue;
                }
                let row = &layer.w[i * layer.n_out..(i + 1) * layer.n_out];
                for (o, &w) in out.iter_mut().zip(row) {
                    *o += hi * w;
                }
            }
            if li < last {
                for o in out.iter_mut() {
                    *o = o.max(0.0);
                }
            }
            h = out;
        }
        // softmax
        let mx = h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut e: Vec<f64> = h.iter().map(|v| (v - mx).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter_mut().for_each(|v| *v /= s);
        let mut proba = vec![0.0; self.classes.iter().max().map_or(0, |m| m + 1)];
        for (j, &c) in self.classes.iter().enumerate() {
            proba[c] = e[j];
        }
        proba
    }
}

/// Индексы нужных признаков в векторе телефона.
pub struct FeatureIdx {
    pub price: usize,
    pub ram: usize,
    pub rom: usize,
    pub battery: usize,
    pub screen: usize,
    pub refresh: usize,
    pub weight: usize,
    pub camera: usize,
    pub antutu: usize,
    pub charge: usize,
    pub g5: usize,
    pub nfc: usize,
    pub esim: usize,
    pub year: usize,
    pub ir: usize,
    pub active: usize,
    pub standby: usize,
}

pub struct Model {
    pub phones: Vec<Phone>,
    pub n: usize,
    pub nf: usize,
    pub k: usize,
    pub features: Vec<String>,
    pub feature_names_ru: Vec<String>,
    pub fi: FeatureIdx,
    pub z: Vec<f64>,
    pub x: Vec<f64>,
    pub labels: Vec<usize>,
    pub clusters: Vec<usize>,
    pub cluster_names: Vec<String>,
    pub class_names: Vec<String>,
    pub members: Vec<Vec<usize>>,
    /// p (n × k) — мягкая принадлежность группам, см. описание модуля
    pub p: Vec<f64>,
    pub rugged: Vec<bool>,
    pub fair: HashMap<i64, f64>,
    pub mlp: Mlp,
    pub trained_at: String,
    pub metrics: serde_json::Value,
    pub ui: serde_json::Value,
    pub python_bench_ms: Option<f64>,
}

pub const SURVEY_FIELDS: [&str; 21] = [
    "budget_rub", "imp_camera", "imp_active", "imp_standby", "imp_perf", "imp_compact", "imp_charge",
    "imp_storage", "imp_new", "imp_value", "imp_rugged", "gaming", "photo", "work", "os_pref", "need_nfc",
    "esim_pref", "ir_pref", "screen_pref", "min_hz", "min_rom",
];

/// «Броневик» — защищённый телефон весом от 300 г (правило из features.py).
pub const RUGGED_WEIGHT_G: f64 = 300.0;

fn flat(m: &[Vec<f64>], cols: usize, what: &str) -> Result<Vec<f64>, String> {
    if m.iter().any(|r| r.len() != cols) {
        return Err(format!("{what}: ожидалось {cols} столбцов"));
    }
    Ok(m.concat())
}

/// Медиана как в numpy (для чётной длины — среднее двух средних элементов).
pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

impl Model {
    pub fn from_json(text: &str) -> Result<Model, String> {
        let raw: RawModel = serde_json::from_str(text).map_err(|e| format!("model.json: {e}"))?;
        if raw.survey_fields != SURVEY_FIELDS {
            return Err("поля анкеты в model.json не совпадают с Rust-версией".into());
        }
        let n = raw.phones.len();
        let nf = raw.features.len();
        let k = raw.cluster_names.len();
        let pos = |name: &str| -> Result<usize, String> {
            raw.features.iter().position(|f| f == name).ok_or(format!("нет признака {name}"))
        };
        let fi = FeatureIdx {
            price: pos("price_rub")?, ram: pos("ram_gb")?, rom: pos("rom_gb")?, battery: pos("battery_mah")?,
            screen: pos("screen_in")?, refresh: pos("refresh_hz")?, weight: pos("weight_g")?,
            camera: pos("camera_mp")?, antutu: pos("antutu_k")?, charge: pos("charge_w")?, g5: pos("has_5g")?,
            nfc: pos("nfc")?, esim: pos("esim")?, year: pos("year")?, ir: pos("ir_port")?,
            active: pos("active_h")?, standby: pos("standby_d")?,
        };
        let z = flat(&raw.z, nf, "z")?;
        let x = flat(&raw.x, nf, "x")?;
        let centers = flat(&raw.centers, nf, "centers")?;
        if raw.z.len() != n || raw.x.len() != n || raw.labels.len() != n || raw.clusters.len() != n {
            return Err("размеры массивов не совпадают с числом телефонов".into());
        }
        let mut members = vec![Vec::new(); k];
        for (i, &c) in raw.clusters.iter().enumerate() {
            members.get_mut(c).ok_or("номер группы вне диапазона")?.push(i);
        }

        // p_ij = softmax_j(−(d²_ij − min_j d²_ij) / τ), τ = медиана(min_j d²_ij) — как soft_cluster_bonus
        let mut d2 = vec![0.0; n * k];
        for i in 0..n {
            for j in 0..k {
                d2[i * k + j] = (0..nf).map(|f| (x[i * nf + f] - centers[j * nf + f]).powi(2)).sum();
            }
        }
        let mins: Vec<f64> = (0..n)
            .map(|i| d2[i * k..(i + 1) * k].iter().cloned().fold(f64::INFINITY, f64::min))
            .collect();
        let tau = median(&mins) + 1e-9;
        let mut p = vec![0.0; n * k];
        for i in 0..n {
            let row = &mut p[i * k..(i + 1) * k];
            for j in 0..k {
                row[j] = (-(d2[i * k + j] - mins[i]) / tau).exp();
            }
            let s: f64 = row.iter().sum();
            row.iter_mut().for_each(|v| *v /= s);
        }

        let layers = raw
            .mlp
            .coefs
            .iter()
            .zip(&raw.mlp.intercepts)
            .map(|(w, b)| Layer { n_in: w.len(), n_out: b.len(), w: w.concat(), b: b.clone() })
            .collect();
        let fair = raw
            .fair
            .iter()
            .filter_map(|(id, v)| id.parse::<i64>().ok().map(|id| (id, *v)))
            .collect();
        let rugged = raw.phones.iter().map(|p| p.weight_g >= RUGGED_WEIGHT_G).collect();
        Ok(Model {
            n, nf, k, fi, z, x, p, members, rugged, fair,
            features: raw.features,
            feature_names_ru: raw.feature_names_ru,
            labels: raw.labels,
            clusters: raw.clusters,
            cluster_names: raw.cluster_names,
            class_names: raw.class_names,
            phones: raw.phones,
            mlp: Mlp {
                mean: raw.mlp.scaler_mean,
                scale: raw.mlp.scaler_scale,
                layers,
                classes: raw.mlp.classes,
            },
            trained_at: raw.trained_at,
            metrics: raw.metrics,
            ui: raw.ui,
            python_bench_ms: raw.python_bench_ms,
        })
    }
}
