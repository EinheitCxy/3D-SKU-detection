import re
from pathlib import Path

import pytest
import requests

from docker.cos_upload import (
    CosUploadConfig,
    _authorization,
    upload_mapping_results,
    validate_taskid,
)


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.raise_calls = 0

    def raise_for_status(self):
        self.raise_calls += 1
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, status_code=200):
        self.calls = []
        self.status_code = status_code

    def put(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.status_code)


def _write_env(path: Path, **values: str) -> Path:
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()))
    return path


def test_from_env_reads_temp_file_and_uploads_prefixed_urls(tmp_path):
    config = CosUploadConfig.from_env(
        _write_env(
            tmp_path / ".env",
            COS_SECRET_ID="id",
            COS_SECRET_KEY="secret",
            COS_BUCKET="bucket",
            COS_REGION="region",
            COS_KEY_PREFIX="mapping-artifacts",
            IGNORED="value",
        )
    )
    session = FakeSession()

    result = upload_mapping_results(
        "task-01", b"{}", b"PK", config, session=session, now=lambda: 1_700_000_000
    )

    assert len(session.calls) == 2
    assert [call[0] for call in session.calls] == [
        "https://bucket.cos.region.myqcloud.com/mapping-artifacts/task-01/global_skus.json",
        "https://bucket.cos.region.myqcloud.com/mapping-artifacts/task-01/viewer_bundle.zip",
    ]
    assert result == {
        "global_skus_url": session.calls[0][0],
        "viewer_bundle_url": session.calls[1][0],
    }
    assert session.calls[0][1]["data"] == b"{}"
    assert session.calls[0][1]["headers"]["Content-Type"] == "application/json"
    assert session.calls[1][1]["data"] == b"PK"
    assert session.calls[1][1]["headers"]["Content-Type"] == "application/zip"
    for url, kwargs in session.calls:
        assert "task-01/" in url
        assert kwargs["timeout"] == (5, 30)
        assert kwargs["allow_redirects"] is False
        assert kwargs["headers"]["Authorization"]
        assert "secret" not in kwargs["headers"]["Authorization"]
        assert kwargs["headers"]["Host"] == "bucket.cos.region.myqcloud.com"


@pytest.mark.parametrize(
    "taskID", ["", "../x", "a/b", "a\\b", " space", 1, None, "x" * 129]
)
def test_invalid_taskID_rejected_before_network(taskID):
    config = CosUploadConfig("id", "key", "bucket", "region", "prefix")
    session = FakeSession()
    with pytest.raises(ValueError):
        upload_mapping_results(taskID, b"{}", b"PK", config, session=session)
    assert session.calls == []


def test_taskID_validation_accepts_allowed_pattern():
    assert validate_taskid("A0._-") == "A0._-"
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", validate_taskid("abc"))


def test_cos_failure_propagates_to_caller():
    config = CosUploadConfig("id", "secret", "bucket", "region", "prefix")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        upload_mapping_results(
            "task",
            b"{}",
            b"PK",
            config,
            session=FakeSession(500),
            now=lambda: 1_700_000_000,
        )


def test_cos_v5_authorization_fixed_clock_regression():
    config = CosUploadConfig(
        "AKIDEXAMPLE", "example-secret-key", "examplebucket", "ap-shanghai", "prefix"
    )
    assert _authorization(
        config, "/task/global_skus.json", "application/json", 1_700_000_000
    ) == (
        "q-sign-algorithm=sha1&q-ak=AKIDEXAMPLE&q-sign-time=1700000000;1700000900&"
        "q-key-time=1700000000;1700000900&q-header-list=content-type;host&"
        "q-url-param-list=&q-signature=18c2a91776162c925f7754f07437d7b53e88cd0e"
    )


@pytest.mark.parametrize("missing_key", ["COS_SECRET_ID", "COS_SECRET_KEY", "COS_BUCKET", "COS_REGION", "COS_KEY_PREFIX"])
def test_missing_required_key_is_rejected_before_network(tmp_path, missing_key):
    values = {
        "COS_SECRET_ID": "id",
        "COS_SECRET_KEY": "key",
        "COS_BUCKET": "bucket",
        "COS_REGION": "region",
        "COS_KEY_PREFIX": "prefix",
    }
    del values[missing_key]
    with pytest.raises(ValueError):
        CosUploadConfig.from_env(_write_env(tmp_path / ".env", **values))


def test_default_env_path_is_loaded(monkeypatch, tmp_path):
    import docker.cos_upload as cos_upload

    default_env = _write_env(
        tmp_path / ".env",
        COS_SECRET_ID="id",
        COS_SECRET_KEY="key",
        COS_BUCKET="bucket",
        COS_REGION="region",
        COS_KEY_PREFIX="prefix",
    )
    monkeypatch.setattr(cos_upload, "__file__", str(default_env.with_name("cos_upload.py")))
    config = CosUploadConfig.from_env()
    assert (config.bucket, config.region, config.key_prefix) == ("bucket", "region", "prefix")


def test_missing_env_file_preserves_os_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        CosUploadConfig.from_env(tmp_path / ".env")


def test_env_file_rejects_non_key_value_line(tmp_path):
    env_file = _write_env(
        tmp_path / ".env",
        COS_SECRET_ID="id",
        COS_SECRET_KEY="key",
        COS_BUCKET="bucket",
        COS_REGION="region",
        COS_KEY_PREFIX="prefix",
    )
    env_file.write_text(env_file.read_text() + "\nbad key=value\n")
    with pytest.raises(ValueError):
        CosUploadConfig.from_env(env_file)


@pytest.mark.parametrize("prefix", ["", "/prefix", "prefix/", "a//b", "a/../b", "a/b c", "a/" + "x" * 129])
def test_invalid_key_prefix_is_rejected_before_network(tmp_path, prefix):
    env_file = _write_env(
        tmp_path / ".env",
        COS_SECRET_ID="id",
        COS_SECRET_KEY="key",
        COS_BUCKET="bucket",
        COS_REGION="region",
        COS_KEY_PREFIX=prefix,
    )
    with pytest.raises(ValueError):
        CosUploadConfig.from_env(env_file)


def test_redirect_is_a_failure():
    config = CosUploadConfig("id", "key", "bucket", "region", "prefix")
    with pytest.raises(requests.HTTPError, match="HTTP 302"):
        upload_mapping_results(
            "task",
            b"{}",
            b"PK",
            config,
            session=FakeSession(302),
            now=lambda: 1_700_000_000,
        )
