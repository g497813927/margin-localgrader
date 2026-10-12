"""Network-free tests of public-weight range integrity and resumable promotion."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import setup_explanation_candidates as downloader


def test_qwen36_download_resolves_pinned_release_and_project_location(tmp_path):
    spec = downloader.CANDIDATES["qwen36"]
    final, state_dir, url, identity = downloader.spec_paths("qwen36", spec, tmp_path)
    assert final == tmp_path / "models/qwen36-35b-a3b-ud-q4-k-xl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    assert state_dir.parent == final.parent
    assert url == "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    assert identity["size"] == 22360456160
    assert identity["sha256"] == "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450"


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    data = b'abcdefghijklmn'
    spec = {'repo': 'Owner/Model', 'revision': 'a' * 40, 'path': 'models/example/model.gguf',
            'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'alias': 'example'}
    monkeypatch.setattr(downloader, 'CANDIDATES', {'pro': spec})
    monkeypatch.setattr(downloader, 'CHUNK_BYTES', 4)
    monkeypatch.setattr(downloader, 'DISK_MARGIN', 0)
    return tmp_path, spec, data


def response(status, body=b'', headers=None):
    return httpx.Response(status, headers=headers or {}, stream=httpx.ByteStream(body))


@contextmanager
def ranges(data, mutate=None):
    requests = []
    def handler(request):
        requests.append(request)
        start, end = map(int, request.headers['range'].removeprefix('bytes=').split('-'))
        assert request.headers['accept-encoding'] == 'identity'
        assert 'authorization' not in request.headers
        assert not request.url.username and not request.url.password
        if mutate:
            changed = mutate(request, start, end)
            if changed is not None:
                return changed
        return response(206, data[start:end+1], {'content-range': f'bytes {start}-{end}/{len(data)}'})
    with httpx.Client(transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False) as client:
        yield client, requests


def test_parallel_download_promotes_only_verified_complete_file(pinned):
    root, spec, data = pinned
    progress = root / 'progress.json'
    with ranges(data) as (client, requests):
        final = downloader.download_candidate('pro', root=root, workers=3, progress_path=progress, client=client)
    assert final.read_bytes() == data
    assert len(requests) == 4
    state_dir = final.parent / ('.' + final.name + '.parallel-download')
    state = json.loads((state_dir / 'state.json').read_text())
    assert len(state['completed']) == 4
    assert not (state_dir / 'weights.part').exists()
    saved = json.loads(progress.read_text())
    assert saved['status'] == 'verified' and saved['verified_bytes'] == len(data)
    assert 'https' not in progress.read_text() and 'url' not in saved


def saved_partial(root, spec, data, *, corrupt=False):
    final, directory, _, identity = downloader.spec_paths('pro', spec, root)
    directory.mkdir(parents=True)
    part = directory / 'weights.part'
    part.write_bytes((b'WRNG' if corrupt else data[:4]) + b'\0' * (len(data)-4))
    downloader.atomic_json(directory / 'state.json', {**identity, 'completed': {'0': hashlib.sha256(data[:4]).hexdigest()}})
    return final, directory, part


@pytest.mark.parametrize('corrupt', [False, True])
def test_resume_rechecks_saved_chunk_hashes(pinned, corrupt):
    root, spec, data = pinned
    final, _, _ = saved_partial(root, spec, data, corrupt=corrupt)
    with ranges(data) as (client, requests):
        downloader.download_candidate('pro', root=root, workers=2, client=client)
    starts = {int(request.headers['range'].split('=')[1].split('-')[0]) for request in requests}
    assert (0 in starts) is corrupt
    assert final.read_bytes() == data


@pytest.mark.parametrize('bad', ['ignored', 'wrong_range', 'wrong_total', 'short', 'oversize', 'length', 'encoding'])
def test_rejects_invalid_http_ranges(pinned, bad):
    root, spec, data = pinned
    def mutate(request, start, end):
        headers = {'content-range': f'bytes {start}-{end}/{len(data)}'}
        body, status = data[start:end+1], 206
        if bad == 'ignored': status = 200
        elif bad == 'wrong_range': headers['content-range'] = f'bytes {start+1}-{end}/{len(data)}'
        elif bad == 'wrong_total': headers['content-range'] = f'bytes {start}-{end}/{len(data)+1}'
        elif bad == 'short': body = body[:-1]
        elif bad == 'oversize': body += b'x'
        elif bad == 'length': headers['content-length'] = '999'
        elif bad == 'encoding': headers['content-encoding'] = 'gzip'
        return response(status, body, headers)
    with ranges(data, mutate) as (client, _):
        with pytest.raises(downloader.DownloadError):
            downloader.download_candidate('pro', root=root, workers=1, client=client)
    assert not (root / spec['path']).exists()


def test_checksum_failure_never_promotes_wrong_weights(pinned):
    root, spec, data = pinned
    with ranges(b'x' * len(data)) as (client, _):
        with pytest.raises(downloader.DownloadError, match='pinned SHA-256'):
            downloader.download_candidate('pro', root=root, client=client)
    final, directory, _, _ = downloader.spec_paths('pro', spec, root)
    assert not final.exists() and (directory / 'weights.part').exists()


def test_checksum_failure_can_recover_on_explicit_rerun(pinned):
    root, spec, data = pinned
    wrong_data = b'x' * len(data)
    with ranges(wrong_data) as (client, requests):
        with pytest.raises(downloader.DownloadError, match='pinned SHA-256'):
            downloader.download_candidate('pro', root=root, workers=2, client=client)
        assert len(requests) == 4
    final, directory, _, identity = downloader.spec_paths('pro', spec, root)
    assert not final.exists()
    assert (directory / 'weights.part').read_bytes() == wrong_data
    assert json.loads((directory / 'state.json').read_text()) == {**identity, 'completed': {}}
    with ranges(data) as (client, requests):
        assert downloader.download_candidate('pro', root=root, workers=2, client=client) == final
        assert len(requests) == 4
    assert final.read_bytes() == data


@pytest.mark.parametrize('location', ['http://cdn-lfs.huggingface.co/file', 'https://evil.example/file', 'https://huggingface.co.evil.example/file', 'https://user:pass@huggingface.co/file'])
def test_unsafe_redirect_not_followed(location):
    requests = []
    def handler(request):
        requests.append(request)
        return response(302, headers={'location': location})
    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        with pytest.raises(downloader.DownloadError, match='HTTPS Hugging Face'):
            downloader.fetch_range(client, 'https://huggingface.co/a/b/resolve/revision/file', 0, 3, 4)
    assert len(requests) == 1


def test_checked_cdn_redirect_preserves_range_without_auth():
    requests = []
    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return response(302, headers={'location': 'https://cas-bridge.xethub.hf.co/public?Signature=never-log-me'})
        assert request.headers['range'] == 'bytes=2-5' and 'authorization' not in request.headers
        return response(206, b'cdef', {'content-range': 'bytes 2-5/9'})
    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        assert downloader.fetch_range(client, 'https://huggingface.co/a/b/file', 2, 5, 9) == b'cdef'


def test_wrong_existing_final_is_left_untouched(pinned):
    root, spec, data = pinned
    final = root / spec['path']; final.parent.mkdir(parents=True); final.write_bytes(b'wrong')
    with ranges(data) as (client, requests):
        with pytest.raises(downloader.DownloadError, match='left untouched'):
            downloader.download_candidate('pro', root=root, client=client)
    assert final.read_bytes() == b'wrong' and not requests


def test_offline_only_verifies_complete_model(pinned):
    root, spec, data = pinned
    with ranges(data) as (client, requests):
        with pytest.raises(downloader.DownloadError, match='Offline mode'):
            downloader.download_candidate('pro', root=root, offline=True, client=client)
        assert not (root / 'models').exists()
        final = root / spec['path']; final.parent.mkdir(parents=True); final.write_bytes(data)
        assert downloader.download_candidate('pro', root=root, offline=True, client=client) == final
        assert not requests


def test_state_for_different_pin_is_not_overwritten(pinned):
    root, spec, data = pinned
    _, directory, part = saved_partial(root, spec, data)
    state_file = directory / 'state.json'
    state = json.loads(state_file.read_text()); state['revision'] = 'b' * 40
    state_file.write_text(json.dumps(state)); original = state_file.read_bytes(), part.read_bytes()
    with ranges(data) as (client, requests):
        with pytest.raises(downloader.DownloadError, match='different model metadata'):
            downloader.download_candidate('pro', root=root, client=client)
    assert original == (state_file.read_bytes(), part.read_bytes()) and not requests


def test_two_writers_cannot_share_state(tmp_path):
    lock = tmp_path / 'writer.lock'
    with downloader.download_lock(lock):
        with pytest.raises(downloader.DownloadError, match='Another parallel downloader'):
            with downloader.download_lock(lock):
                pytest.fail('second writer acquired the lock')


def test_not_enough_space_prevents_requests(pinned, monkeypatch):
    root, spec, data = pinned
    monkeypatch.setattr(downloader.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1))
    with ranges(data) as (client, requests):
        with pytest.raises(downloader.DownloadError, match='free disk space'):
            downloader.download_candidate('pro', root=root, client=client)
    assert not requests and not (root / spec['path']).exists()


def test_symlinked_model_directory_is_rejected(pinned):
    root, spec, data = pinned
    (root / 'models').mkdir(); target = root / 'other'; target.mkdir()
    (root / 'models/example').symlink_to(target, target_is_directory=True)
    with pytest.raises(downloader.DownloadError, match='symbolic links'):
        downloader.download_candidate('pro', root=root, offline=True)


def test_unrelated_progress_file_is_not_overwritten(pinned):
    root, _, data = pinned
    progress = root / 'unrelated.json'; progress.write_text('{"important":"keep"}')
    with ranges(data) as (client, requests):
        with pytest.raises(downloader.DownloadError, match='unrelated data'):
            downloader.download_candidate('pro', root=root, client=client, progress_path=progress)
    assert progress.read_text() == '{"important":"keep"}' and not requests


@pytest.mark.parametrize('value', [0, 17, 'x'])
def test_worker_limits(value):
    with pytest.raises(Exception, match='between 1 and 16'):
        downloader.workers_value(value)
