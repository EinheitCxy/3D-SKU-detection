import re
from pathlib import Path

import pytest

import docker.cos_upload as cos_upload

from docker.cos_upload import (
    CosUploadConfig,
    upload_viewer_bundle,
    validate_taskid,
)


class FakeCosClient:
    def __init__(self, failure: Exception | None = None):
        self.calls = []
        self.failure = failure

    def put_object(self, **kwargs):
        self.calls.append(("put_object", kwargs))
        if self.failure is not None:
            raise self.failure
        return {}

    def get_presigned_url(self, **kwargs):
        self.calls.append(("get_presigned_url", kwargs))
        return "https://signed.example/viewer_bundle.zip"


def _write_env(path: Path, **values: str) -> Path:
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()))
    return path


def test_from_env_reads_temp_file_and_uploads_prefixed_viewer_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
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
    client = FakeCosClient()
    monkeypatch.setattr(cos_upload, "CosS3Client", lambda _config: client)

    result = upload_viewer_bundle("task-01", b"PK", config)

    assert client.calls == [
        ("put_object", {
            "Bucket": "bucket",
            "Key": "mapping-artifacts/task-01/viewer_bundle.zip",
            "Body": b"PK",
            "ContentType": "application/zip",
        }),
    ]
    assert result == {
        "viewer_bundle_url": "https://bucket.cos.region.myqcloud.com/mapping-artifacts/task-01/viewer_bundle.zip",
    }


@pytest.mark.parametrize(
    "taskID", ["", "../x", "a/b", "a\\b", " space", 1, None, "x" * 129]
)
def test_invalid_taskID_rejected_before_network(taskID):
    config = CosUploadConfig("id", "key", "bucket", "region", "prefix")
    with pytest.raises(ValueError):
        upload_viewer_bundle(taskID, b"PK", config)


def test_taskID_validation_accepts_allowed_pattern():
    assert validate_taskid("A0._-") == "A0._-"
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", validate_taskid("abc"))


def test_cos_sdk_failure_propagates_to_caller(monkeypatch: pytest.MonkeyPatch):
    config = CosUploadConfig("id", "secret", "bucket", "region", "prefix")
    monkeypatch.setattr(
        cos_upload, "CosS3Client", lambda _config: FakeCosClient(RuntimeError("SDK failed"))
    )
    with pytest.raises(RuntimeError, match="SDK failed"):
        upload_viewer_bundle("task", b"PK", config)


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
