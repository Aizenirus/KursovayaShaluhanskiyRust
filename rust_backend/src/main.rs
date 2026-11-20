//! HTTP-сервер подбора смартфона.
//!
//!   GET  /                — веб-страница (static/index.html, встроена в бинарник)
//!   GET  /api/meta        — справочник для интерфейса: ползунки, пресеты, метрики модели
//!   POST /api/recommend   — подбор по анкете (JSON) → выдача, блоки, сегменты, время расчёта
//!   GET  /api/bench?n=…   — замер скорости на n случайных анкетах
//!   GET  /fonts/…         — шрифты страницы (встроены в бинарник)
//!
//! Переменные окружения: PORT (по умолчанию 8080), MODEL_PATH (иначе — встроенный model.json).

use axum::{
    extract::{Path, Query, State},
    http::header,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use phonepicker_server::{bench, engine, model::Model};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
use tower_http::{compression::CompressionLayer, cors::CorsLayer};

const INDEX_HTML: &str = include_str!("../static/index.html");
const EMBEDDED_MODEL: &str = include_str!("../model.json");

/// Шрифты Unbounded и Onest (SIL OFL) встроены в бинарник: страница выглядит одинаково и без интернета.
macro_rules! fonts {
    ($($name:literal),* $(,)?) => { &[$(($name, include_bytes!(concat!("../static/fonts/", $name)) as &[u8])),*] };
}
const FONTS: &[(&str, &[u8])] = fonts![
    "unbounded-cyrillic-500-normal.woff2", "unbounded-latin-500-normal.woff2",
    "unbounded-cyrillic-700-normal.woff2", "unbounded-latin-700-normal.woff2",
    "unbounded-cyrillic-800-normal.woff2", "unbounded-latin-800-normal.woff2",
    "onest-cyrillic-400-normal.woff2", "onest-latin-400-normal.woff2",
    "onest-cyrillic-500-normal.woff2", "onest-latin-500-normal.woff2",
    "onest-cyrillic-600-normal.woff2", "onest-latin-600-normal.woff2",
    "onest-cyrillic-700-normal.woff2", "onest-latin-700-normal.woff2",
];

#[derive(Deserialize)]
struct RecommendRequest {
    #[serde(flatten)]
    survey: engine::Survey,
    #[serde(default = "default_top")]
    top_n: usize,
    #[serde(default = "yes")]
    diverse: bool,
}
fn default_top() -> usize {
    10
}
fn yes() -> bool {
    true
}

#[derive(Serialize)]
struct RecommendResponse {
    #[serde(flatten)]
    result: engine::Recommendation,
    segments: Vec<engine::Segment>,
    total: usize,
    compute_us: f64,
}

async fn recommend(State(m): State<Arc<Model>>, Json(req): Json<RecommendRequest>) -> Json<RecommendResponse> {
    let t0 = Instant::now();
    let a = req.survey.sanitize();
    let top_n = req.top_n.clamp(1, 50);
    let result = engine::recommend(&m, &a, top_n, req.diverse, true);
    let segments = engine::recommend_by_segments(&m, &a, top_n, req.diverse);
    let compute_us = t0.elapsed().as_secs_f64() * 1e6;
    Json(RecommendResponse { result, segments, total: m.n, compute_us })
}

async fn meta(State(m): State<Arc<Model>>) -> impl IntoResponse {
    let mut sizes = vec![0usize; m.k];
    m.clusters.iter().for_each(|&c| sizes[c] += 1);
    let groups: Vec<_> = m
        .cluster_names
        .iter()
        .zip(&sizes)
        .map(|(n, s)| serde_json::json!({"name": n, "size": s}))
        .collect();
    let (pmin, pmax) = m.phones.iter().fold((f64::MAX, 0.0f64), |(a, b), p| (a.min(p.price_rub), b.max(p.price_rub)));
    let mut brands: Vec<&str> = m.phones.iter().map(|p| p.brand.as_str()).collect();
    brands.sort();
    brands.dedup();
    Json(serde_json::json!({
        "phones": m.n, "groups": groups, "classes": m.class_names, "features": m.nf,
        "metrics": m.metrics, "ui": m.ui, "trained_at": m.trained_at,
        "price_min": pmin, "price_max": pmax, "python_bench_ms": m.python_bench_ms,
        "brands": brands.len(),
    }))
}

#[derive(Deserialize)]
struct BenchQuery {
    n: Option<usize>,
}

async fn bench_handler(State(m): State<Arc<Model>>, Query(q): Query<BenchQuery>) -> impl IntoResponse {
    let n = q.n.unwrap_or(2000).clamp(10, 50_000);
    let m2 = m.clone();
    let res = tokio::task::spawn_blocking(move || {
        let mut rng = bench::Rng::new(42);
        let surveys: Vec<_> = (0..n).map(|_| bench::random_survey(&mut rng)).collect();
        let mut times = Vec::with_capacity(n);
        let t_all = Instant::now();
        for a in &surveys {
            let t = Instant::now();
            let r = engine::recommend(&m2, a, 10, true, true);
            std::hint::black_box(&r);
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let total = t_all.elapsed().as_secs_f64();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
        serde_json::json!({
            "n": n, "mean_us": times.iter().sum::<f64>() / n as f64,
            "p50_us": pct(0.5), "p99_us": pct(0.99), "per_second": n as f64 / total,
            "python_ms": m2.python_bench_ms,
        })
    })
    .await
    .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}));
    Json(res)
}

async fn font(Path(name): Path<String>) -> axum::response::Response {
    match FONTS.iter().find(|(n, _)| *n == name) {
        Some((_, bytes)) => (
            [(header::CONTENT_TYPE, "font/woff2"), (header::CACHE_CONTROL, "public, max-age=31536000, immutable")],
            *bytes,
        )
            .into_response(),
        None => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

async fn index() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-cache")], Html(INDEX_HTML))
}

#[tokio::main]
async fn main() {
    let text = match std::env::var("MODEL_PATH") {
        Ok(path) => std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("не читается {path}: {e}")),
        Err(_) => EMBEDDED_MODEL.to_string(),
    };
    let model = Arc::new(Model::from_json(&text).unwrap_or_else(|e| panic!("ошибка модели: {e}")));
    println!("модель: {} телефонов, {} групп, обучена {}", model.n, model.k, model.trained_at);

    let app = Router::new()
        .route("/", get(index))
        .route("/api/meta", get(meta))
        .route("/api/recommend", post(recommend))
        .route("/api/bench", get(bench_handler))
        .route("/fonts/{name}", get(font))
        .layer(CorsLayer::permissive()) // страницу можно открыть отдельно (файлом или с другого хоста)
        .layer(CompressionLayer::new())
        .with_state(model);

    let port = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080u16);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.expect("порт занят");
    println!("PhonePicker: http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}
