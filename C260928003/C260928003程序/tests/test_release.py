import io
from pathlib import Path

import pytest
from PIL import Image

import app as module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'OUTPUT_DIR', tmp_path / 'outputs')
    return module.create_app().test_client()


def test_only_final_tasks_are_available(client):
    response = client.get('/')
    assert response.status_code == 200
    assert '小麦与大麦'.encode() in response.data
    assert '燕麦'.encode() not in response.data
    health = client.get('/api/health').get_json()
    assert health['status'] == 'ok'
    assert health['models'] == {'wheat': True, 'growth': True, 'lodging': True}
    assert client.post('/api/predict-oat-growth').status_code == 404
    assert client.get('/annotation').status_code == 404
    assert client.post('/api/predict', data={'mode': 'species'}).status_code == 400
    assert client.post('/api/predict').status_code == 400
    assert client.get('/api/download/bad/mask').status_code == 400


@pytest.mark.parametrize('mode', ['wheat', 'growth', 'lodging'])
def test_actual_models_prediction_download_history_and_delete(client, mode):
    root = module.PROJECT_DIR
    sample = (next((root / 'wheat_head_dataset_500/images').iterdir()) if mode == 'wheat'
              else root.parent / '系统测试样例/大麦五通道测试.tif')
    count = 2 if mode == 'wheat' else 1
    if mode == 'wheat':
        with Image.open(sample) as image:
            expected_size = image.size
    else:
        height, width, _ = module.load_five_band(sample).shape
        expected_size = width, height
    response = client.post('/api/predict', data={'mode': mode, 'images': [
        (io.BytesIO(sample.read_bytes()), f'sample_{index}{sample.suffix}') for index in range(count)
    ]}, content_type='multipart/form-data')
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body['image_count'] == count
    csv_response = client.get(body['batch_download_url'])
    assert csv_response.status_code == 200
    assert 'sample_0' in csv_response.data.decode('utf-8-sig')
    csv_response.close()
    for result in body['results']:
        assert result['mode'] == mode
        assert 0 <= result['positive_coverage'] <= 1
        if mode == 'wheat':
            assert result['confidence'] is None
        else:
            assert 0 <= result['confidence'] <= 1
        for key in ['preview_url', 'mask_url', 'overlay_url', 'mask_download_url', 'overlay_download_url']:
            output = client.get(result[key])
            assert output.status_code == 200
            with Image.open(io.BytesIO(output.data)) as image:
                assert image.size == expected_size
            output.close()
    assert client.get('/api/history').get_json()['count'] == count
    for result in body['results']:
        assert client.delete('/api/history/' + result['run_id']).status_code == 200
    assert client.get('/api/history').get_json()['count'] == 0


def test_corrupt_and_incorrect_channel_images_are_rejected(client):
    assert client.post('/api/predict', data={'mode': 'wheat', 'images': (io.BytesIO(b'broken'), 'bad.jpg')},
                       content_type='multipart/form-data').status_code == 400
    stream = io.BytesIO()
    Image.new('RGB', (40, 40)).save(stream, format='TIFF')
    stream.seek(0)
    assert client.post('/api/predict', data={'mode': 'growth', 'images': (stream, 'rgb.tif')},
                       content_type='multipart/form-data').status_code == 400
    assert client.get('/api/history').get_json()['count'] == 0
