PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL CHECK (platform IN ('jd', 'tmall', 'manual', 'third_party')),
    source_site TEXT,
    platform_product_id TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('keyboard', 'mouse')),
    brand TEXT NOT NULL,
    model TEXT,
    product_name TEXT NOT NULL,
    shop_name TEXT,
    product_url TEXT,
    image_url TEXT,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (platform, platform_product_id)
);

CREATE INDEX IF NOT EXISTS idx_products_category_brand
ON products (category, brand);

CREATE TABLE IF NOT EXISTS product_specs (
    product_id INTEGER PRIMARY KEY,
    connection_type TEXT,
    weight_g REAL,
    polling_rate_hz INTEGER,
    rgb INTEGER CHECK (rgb IS NULL OR rgb IN (0, 1)),
    warranty_months INTEGER,
    dpi_max INTEGER,
    sensor_model TEXT,
    button_count INTEGER,
    handedness TEXT,
    switch_type TEXT,
    switch_name TEXT,
    layout_size TEXT,
    key_count INTEGER,
    actuation_mm REAL,
    hot_swappable INTEGER CHECK (hot_swappable IS NULL OR hot_swappable IN (0, 1)),
    raw_specs_json TEXT,
    source_text TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    list_price REAL,
    sale_price REAL NOT NULL CHECK (sale_price > 0),
    coupon_price REAL,
    currency TEXT NOT NULL DEFAULT 'CNY',
    promotion_text TEXT,
    price_source TEXT NOT NULL DEFAULT 'unspecified',
    collected_at TEXT NOT NULL,
    collected_date TEXT NOT NULL,
    source_batch_id TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    UNIQUE (product_id, collected_date)
);

CREATE INDEX IF NOT EXISTS idx_price_product_time
ON price_snapshots (product_id, collected_at);

CREATE TABLE IF NOT EXISTS sales_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    sales_value REAL,
    metric_type TEXT NOT NULL CHECK (metric_type IN ('actual', 'estimated', 'proxy')),
    metric_name TEXT NOT NULL,
    review_count INTEGER,
    rating REAL,
    ranking INTEGER,
    raw_metric_text TEXT,
    source_url TEXT,
    source_payload_json TEXT,
    collected_at TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    source_review_id TEXT,
    rating REAL,
    content TEXT NOT NULL,
    reviewed_at TEXT,
    variant_text TEXT,
    source_product_id TEXT,
    sample_group TEXT CHECK (sample_group IS NULL OR sample_group IN ('negative', 'neutral', 'positive')),
    source_url TEXT,
    collected_at TEXT NOT NULL,
    is_valid INTEGER NOT NULL DEFAULT 1 CHECK (is_valid IN (0, 1)),
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    UNIQUE (product_id, source_review_id)
);

CREATE TABLE IF NOT EXISTS review_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    tag_name TEXT NOT NULL,
    tag_count INTEGER,
    raw_count_text TEXT,
    source_url TEXT,
    collected_at TEXT NOT NULL,
    collected_date TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    UNIQUE (product_id, tag_name, collected_date)
);

CREATE INDEX IF NOT EXISTS idx_review_tags_name
ON review_tags (tag_name);

CREATE TABLE IF NOT EXISTS preference_scores (
    product_id INTEGER NOT NULL,
    score_version TEXT NOT NULL,
    feel_score REAL NOT NULL CHECK (feel_score BETWEEN 0 AND 100),
    response_score REAL NOT NULL CHECK (response_score BETWEEN 0 AND 100),
    comfort_score REAL NOT NULL CHECK (comfort_score BETWEEN 0 AND 100),
    durability_score REAL NOT NULL CHECK (durability_score BETWEEN 0 AND 100),
    appearance_score REAL NOT NULL CHECK (appearance_score BETWEEN 0 AND 100),
    value_score REAL NOT NULL CHECK (value_score BETWEEN 0 AND 100),
    parameter_score_json TEXT NOT NULL,
    review_score_json TEXT,
    calculated_at TEXT NOT NULL,
    PRIMARY KEY (product_id, score_version),
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS collection_jobs (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'success', 'partial', 'failed')),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    fetched_count INTEGER NOT NULL DEFAULT 0,
    inserted_count INTEGER NOT NULL DEFAULT 0,
    updated_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT
);

CREATE TABLE IF NOT EXISTS user_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    page_name TEXT NOT NULL,
    event_name TEXT NOT NULL,
    event_data_json TEXT,
    duration_ms INTEGER,
    is_success INTEGER NOT NULL CHECK (is_success IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_annotation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_sku TEXT NOT NULL,
    source_review_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    sentiment TEXT NOT NULL CHECK (sentiment IN ('正向', '中性', '混合', '负向')),
    aspects_json TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('已确认', '已修改', '排除', '需再核对')),
    note TEXT NOT NULL DEFAULT '',
    annotation_origin TEXT NOT NULL DEFAULT 'AI辅助人工复核',
    reviewed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_annotation_identity
ON review_annotation_events (target_sku, source_review_id, id);

CREATE TABLE IF NOT EXISTS product_identity_verifications (
    product_id INTEGER PRIMARY KEY,
    target_sku TEXT NOT NULL,
    model_key TEXT NOT NULL,
    verified_title TEXT NOT NULL,
    verified_url TEXT NOT NULL,
    source_attributes_json TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('verified', 'pending')),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

