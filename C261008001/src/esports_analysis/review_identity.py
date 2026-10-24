"""Resolve known source SKUs without modifying original collection ownership."""
import pandas as pd


def load_resolved_reviews(connection) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT r.id AS review_row_id,
               COALESCE(sp.id,p.id) AS product_id,
               COALESCE(sp.platform_product_id,p.platform_product_id) AS platform_product_id,
               COALESCE(sp.category,p.category) AS category,
               COALESCE(sp.brand,p.brand) AS brand,
               COALESCE(sp.product_name,p.product_name) AS product_name,
               r.source_review_id,r.rating,r.content,r.reviewed_at,r.variant_text,
               r.source_product_id,r.sample_group,
               COALESCE(sp.product_url,r.source_url) AS source_url,r.collected_at,
               p.id AS requested_product_id,p.platform_product_id AS requested_sku,
               r.source_url AS requested_url,
               CASE WHEN sp.id IS NOT NULL THEN '来源SKU商品页已核验'
                    WHEN r.source_product_id=p.platform_product_id THEN '来源SKU与入口一致'
                    ELSE '来源待核验' END AS identity_resolution
        FROM reviews r JOIN products p ON p.id=r.product_id
        LEFT JOIN product_identity_verifications v
            ON v.target_sku=r.source_product_id AND v.status='verified'
        LEFT JOIN products sp ON sp.id=v.product_id AND sp.platform='jd'
        WHERE r.is_valid=1 ORDER BY r.id
    """, connection)


def load_verified_product_catalog(database_path) -> pd.DataFrame:
    from .db import connect, initialize_database
    initialize_database(database_path)
    with connect(database_path) as db:
        return pd.read_sql_query("""
            SELECT p.id AS product_id,p.category,p.brand,p.product_name,
                   p.platform_product_id AS sku,v.model_key,v.verified_at,
                   v.verified_url,COUNT(s.id) AS history_points,MIN(s.sale_price) AS lowest_price
            FROM product_identity_verifications v JOIN products p ON p.id=v.product_id
            LEFT JOIN price_snapshots s ON s.product_id=p.id AND s.price_source='manmanbuy'
            WHERE v.status='verified' GROUP BY p.id ORDER BY p.category,p.brand,p.product_name
        """,db)
