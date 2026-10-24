from __future__ import annotations

import csv
import json
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from barley_inference import load_five_band, load_task_ensemble, predict_map, rgb_preview, visualizations
from predict import DEFAULT_CHECKPOINTS, load_models, make_overlay, predict_image

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / 'web_outputs'
MAX_BATCH_IMAGES = 10
RUN_ID_PATTERN = re.compile(r'^[0-9a-f]{12}$')
MODES = {'wheat': '小麦麦穗分割', 'growth': '大麦长势分类', 'lodging': '大麦倒伏区域识别'}
_MODEL_CACHE = {}


def checkpoint_paths():
    return {'wheat': DEFAULT_CHECKPOINTS,
            **{task: [PROJECT_DIR / 'outputs/spring_barley_multitask_5band' / task / f'seed_{seed}/best.pt'
                      for seed in (42, 123, 2026)] for task in ('growth', 'lodging')}}


def get_models(mode):
    if mode not in _MODEL_CACHE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        paths = checkpoint_paths()[mode]
        models, config = load_models(paths, device) if mode == 'wheat' else load_task_ensemble(paths, device)
        _MODEL_CACHE[mode] = models, config, device
    return _MODEL_CACHE[mode]


def result_urls(run_id):
    return {**{f'{kind}_url': f'/outputs/{run_id}/{kind}.png' for kind in ('preview', 'mask', 'overlay')},
            **{f'{kind}_download_url': f'/api/download/{run_id}/{kind}' for kind in ('mask', 'overlay')}}


def read_summary(run_dir):
    try:
        data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
        data.update(result_urls(data['run_id']))
        return data
    except (OSError, ValueError, KeyError):
        return None


def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 120 * 1024 * 1024
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/health')
    def health():
        ready = {key: all(path.is_file() for path in paths) for key, paths in checkpoint_paths().items()}
        return jsonify(status='ok' if all(ready.values()) else 'degraded', framework='Flask+Vue3',
                       device='cuda' if torch.cuda.is_available() else 'cpu', models=ready,
                       max_batch_images=MAX_BATCH_IMAGES)

    @app.post('/api/predict')
    def predict():
        mode = request.form.get('mode', 'wheat')
        if mode not in MODES:
            return jsonify(error='无效的分析模式'), 400
        uploads = [file for file in request.files.getlist('images') if file.filename]
        if not 1 <= len(uploads) <= MAX_BATCH_IMAGES:
            return jsonify(error=f'请选择1—{MAX_BATCH_IMAGES}个图像文件'), 400
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'} if mode == 'wheat' else {'.tif', '.tiff'}
        if any(Path(file.filename).suffix.lower() not in extensions for file in uploads):
            return jsonify(error='小麦支持PNG/JPG/TIFF，大麦需要五通道TIFF'), 400
        models, config, device = get_models(mode)
        results, batch_started = [], time.perf_counter()
        created = []
        try:
            for uploaded in uploads:
                run_id = uuid.uuid4().hex[:12]
                run_dir = OUTPUT_DIR / run_id
                run_dir.mkdir()
                created.append(run_dir)
                # 输入固定保存为input，避免上传文件名与结果文件冲突。
                suffix = Path(uploaded.filename).suffix.lower()
                input_path = run_dir / ('input' + suffix)
                uploaded.save(input_path)
                started = time.perf_counter()
                if mode == 'wheat':
                    original, mask = predict_image(input_path, models, config, device)
                    original.save(run_dir / 'preview.png')
                    from PIL import Image
                    Image.fromarray((mask * 255).astype(np.uint8)).save(run_dir / 'mask.png')
                    make_overlay(original, mask, 0.45).save(run_dir / 'overlay.png')
                    prediction = dict(label='wheat_head', label_zh='麦穗分割完成', confidence=None,
                                      positive_coverage=float((mask == 1).mean()))
                    width, height = original.size
                    channels = 3
                else:
                    image = load_five_band(input_path)
                    rgb_preview(image).save(run_dir / 'preview.png')
                    prediction, heatmap, mask = predict_map(image, models, config, device)
                    mask_image, overlay = visualizations(image, heatmap, mask)
                    mask_image.save(run_dir / 'mask.png')
                    overlay.save(run_dir / 'overlay.png')
                    labels = {'normal_vital': '正常/旺盛', 'dry': '干枯/衰退', 'standing': '正常直立', 'lodged': '倒伏'}
                    prediction['label_zh'] = labels[prediction['label']]
                    height, width, channels = image.shape
                    if height % 30 or width % 30:
                        prediction['warning'] = '倒伏/长势区域按30×30图块预测；大于30像素的图像不足一块的右侧、下侧边缘不参与预测。'
                input_path.unlink()
                summary = dict(run_id=run_id, created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                               original_name=secure_filename(uploaded.filename) or ('image' + suffix),
                               mode=mode, mode_zh=MODES[mode], width=int(width), height=int(height), channels=channels,
                               elapsed=time.perf_counter()-started, device=str(device), model_count=3, **prediction)
                (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
                results.append({**summary, **result_urls(run_id)})
        except (ValueError, OSError) as error:
            for run_dir in created:
                shutil.rmtree(run_dir)
            return jsonify(error=f'无法读取图像或输入通道不符合要求：{error}'), 400
        batch_id = results[0]['run_id']
        fields = ['run_id', 'original_name', 'mode_zh', 'label_zh', 'confidence', 'positive_coverage', 'elapsed']
        with (OUTPUT_DIR / batch_id / 'batch_predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        download_url = f'/api/download/{batch_id}/batch_predictions'
        summary_path = OUTPUT_DIR / batch_id / 'summary.json'
        first = json.loads(summary_path.read_text(encoding='utf-8'))
        first['batch_download_url'] = download_url
        summary_path.write_text(json.dumps(first, ensure_ascii=False, indent=2), encoding='utf-8')
        return jsonify(results=results, image_count=len(results), batch_elapsed=time.perf_counter()-batch_started,
                       batch_download_url=download_url)

    @app.get('/api/history')
    def history():
        items = [read_summary(path) for path in OUTPUT_DIR.iterdir() if path.is_dir()]
        valid = sorted((item for item in items if item), key=lambda item: item['created_at'], reverse=True)
        return jsonify(items=valid[:50], count=len(valid))

    @app.delete('/api/history/<run_id>')
    def delete_history(run_id):
        if not RUN_ID_PATTERN.fullmatch(run_id):
            return jsonify(error='无效的记录编号'), 400
        run_dir = OUTPUT_DIR / run_id
        if not run_dir.is_dir():
            return jsonify(error='记录不存在'), 404
        shutil.rmtree(run_dir)
        return jsonify(deleted=run_id)

    @app.get('/api/download/<run_id>/<kind>')
    def download(run_id, kind):
        if not RUN_ID_PATTERN.fullmatch(run_id) or kind not in {'mask', 'overlay', 'batch_predictions'}:
            return jsonify(error='无效的下载参数'), 400
        filename = 'batch_predictions.csv' if kind == 'batch_predictions' else f'{kind}.png'
        return send_from_directory(OUTPUT_DIR / run_id, filename, as_attachment=True)

    @app.get('/outputs/<run_id>/<path:filename>')
    def output_file(run_id, filename):
        if not RUN_ID_PATTERN.fullmatch(run_id) or filename not in {'mask.png', 'overlay.png', 'preview.png'}:
            return jsonify(error='无效的图片参数'), 400
        return send_from_directory(OUTPUT_DIR / run_id, filename)

    @app.errorhandler(Exception)
    def error_handler(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        app.logger.exception('Request failed')
        return jsonify(error='分析失败，请检查输入文件和模型'), 500

    return app


app = create_app()
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
