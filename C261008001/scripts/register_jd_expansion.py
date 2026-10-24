"""Register only source-number/title verified JD candidates, then import quotes."""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
from esports_analysis.config import get_database_path
from esports_analysis.db import connect, initialize_database
from esports_analysis.importer import import_products_csv
from import_jd_research_data import infer_page_specs, SPEC_FIELDS

MODELS = {
    '100001691967': ('logitech_g502_hero_wired', ['g502','hero']),
    '100041149519': ('logitech_g502x_wireless', ['g502x','无线']),
    '100033870975': ('razer_deathadder_v3_pro', ['蝰蛇','v3','专业']),
    '100190452813': ('razer_deathadder_v4_pro', ['蝰蛇','v4','专业']),
    '100063306539': ('razer_blackwidow_v4', ['黑寡妇','v4']),
    '100043546994': ('razer_blackwidow_v4_pro', ['黑寡妇','v4','专业']),
    '100266391406': ('razer_blackwidow_v4_lowprofile', ['黑寡妇','v4','矮']),
    '100056847840': ('razer_cobra_pro', ['眼镜蛇','专业']),
    '100108649784': ('razer_viper_v3_pro', ['毒蝰','v3','专业']),
}


def main():
    database = get_database_path()
    initialize_database(database)
    raw = []
    for name in ('jd_expansion_probe_20261014.json','jd_expansion_details_20261014.json'):
        path = ROOT / 'data/raw' / name
        if path.exists():
            raw.extend(json.loads(path.read_text(encoding='utf-8')).get('products',[]))
    accepted = []
    rejected = []
    for product in raw:
        sku = str(product['sku'])
        title = product.get('title','')
        attrs = product.get('attributes') or []
        attr_dict = {x.get('labelName'):x.get('labelValue') for x in attrs}
        normalized = re.sub(r'\s+','',title).lower()
        model_key,tokens = MODELS[sku]
        if not product.get('title_from_source') or str(attr_dict.get('商品编号')) != sku:
            rejected.append({'sku':sku,'reason':'商品页编号或源标题无法核验'})
            continue
        if not all(token in normalized for token in tokens):
            rejected.append({'sku':sku,'reason':'源标题与预期版本不匹配','title':title})
            continue
        accepted.append({**product,'model_key':model_key})
    rows = [{
        'platform':'jd','platform_product_id':p['sku'],'category':p['category'],
        'brand':p['brand'],'model':p['model_key'],'product_name':p['title'],
        'shop_name':'京东商品页','product_url':p['product_url'],
        'collected_at':p['collected_at'],'price_source':'jd_verified_catalog',
    } for p in accepted]
    catalog = ROOT / 'data/processed/jd_expansion_verified_catalog_20261014.csv'
    if rows:
        with catalog.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        imported = import_products_csv(catalog,database)
        if imported.failed_count:
            raise RuntimeError('商品目录导入失败')
    with connect(database) as db:
        for product in accepted:
            sku=product['sku']
            pid=db.execute("SELECT id FROM products WHERE platform='jd' AND platform_product_id=?",(sku,)).fetchone()[0]
            db.execute('''INSERT INTO product_identity_verifications (
                product_id,target_sku,model_key,verified_title,verified_url,source_attributes_json,verified_at,status
            ) VALUES (?,?,?,?,?,?,?,'verified') ON CONFLICT(product_id) DO UPDATE SET
                target_sku=excluded.target_sku,model_key=excluded.model_key,
                verified_title=excluded.verified_title,verified_url=excluded.verified_url,
                source_attributes_json=excluded.source_attributes_json,verified_at=excluded.verified_at,status='verified'
            ''',(pid,sku,product['model_key'],product['title'],product['product_url'],
                 json.dumps(product['attributes'],ensure_ascii=False),product['collected_at']))
            db.execute('UPDATE product_specs SET raw_specs_json=?, source_text=?, updated_at=? WHERE product_id=?',
                       (json.dumps(product['attributes'],ensure_ascii=False),product['product_url'],product['collected_at'],pid))
            specs=infer_page_specs(product)
            assignments=', '.join(f'{field}=COALESCE(?, {field})' for field in SPEC_FIELDS)
            db.execute(f'UPDATE product_specs SET {assignments} WHERE product_id=?',
                       (*[specs[field] for field in SPEC_FIELDS],pid))
    quotes_path = ROOT / 'data/processed/jd_expansion_history_20261014.csv'
    imported_quotes=0
    quote_errors=[]
    if quotes_path.exists():
        quotes=list(csv.DictReader(quotes_path.open(encoding='utf-8-sig',newline='')))
        accepted_ids={p['sku']:p for p in accepted}
        verified_quotes=[]
        for row in quotes:
            sku=row['platform_product_id']
            if sku not in accepted_ids:
                continue
            tokens=MODELS[sku][1]
            normalized=re.sub(r'\s+','',row['product_name']).lower()
            if not all(token in normalized for token in tokens):
                quote_errors.append(sku)
                continue
            row['product_name']=accepted_ids[sku]['title']
            verified_quotes.append(row)
        if verified_quotes:
            clean_path=ROOT / 'data/processed/jd_expansion_verified_history_20261014.csv'
            with clean_path.open('w',encoding='utf-8-sig',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(verified_quotes[0]));writer.writeheader();writer.writerows(verified_quotes)
            result=import_products_csv(clean_path,database)
            imported_quotes=result.fetched_count-result.failed_count
    with connect(database) as db:
        resolved=db.execute('''SELECT COUNT(*) FROM reviews r JOIN product_identity_verifications v
                             ON v.target_sku=r.source_product_id AND v.status='verified'
                             WHERE r.source_product_id NOT IN (
                                 SELECT platform_product_id FROM products WHERE id=r.product_id
                             )''').fetchone()[0]
    report={'verified_models':len({p['model_key'] for p in accepted}),
            'verified_skus':[p['sku'] for p in accepted],'rejected':rejected,
            'imported_price_rows':imported_quotes,'quote_title_mismatch_skus':sorted(set(quote_errors)),
            'existing_reviews_resolved_by_source_sku':resolved,
            'note':'仅按核验的来源SKU解析原评论，未改写原评论商品归属，未将商品页共享好评自动视为严格SKU评论'}
    (ROOT / 'logs/jd_expansion_registration_20261014.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
