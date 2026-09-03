"""Upload mapping artifacts to Tencent Cloud COS using COS XML API signatures."""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import requests


_TASKID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_KEY_PREFIX_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SIGNATURE_TTL = 900


@dataclass(frozen=True)
class CosUploadConfig:
    secret_id: str
    secret_key: str
    bucket: str
    region: str
    key_prefix: str

    def __post_init__(self) -> None:
        if not all(
            (self.secret_id, self.secret_key, self.bucket, self.region, self.key_prefix)
        ):
            raise ValueError("COS_SECRET_ID, COS_SECRET_KEY, COS_BUCKET, COS_REGION, and COS_KEY_PREFIX are required")
        validate_key_prefix(self.key_prefix)

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "CosUploadConfig":
        env_file = Path(__file__).with_name(".env") if env_file is None else env_file
        values: dict[str, str] = {}
        lines = env_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError("COS environment file must contain only KEY=VALUE lines")
            key, value = line.split("=", 1)
            if _ENV_KEY_RE.fullmatch(key) is None:
                raise ValueError("COS environment file must contain only KEY=VALUE lines")
            values[key] = value
        required = (
            "COS_SECRET_ID",
            "COS_SECRET_KEY",
            "COS_BUCKET",
            "COS_REGION",
            "COS_KEY_PREFIX",
        )
        missing = [key for key in required if not values.get(key)]
        if missing:
            raise ValueError(f"missing required COS settings: {', '.join(missing)}")
        return cls(*(values[key] for key in required))


def validate_taskid(taskid: object) -> str:
    if not isinstance(taskid, str) or _TASKID_RE.fullmatch(taskid) is None:
        raise ValueError("taskID must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    return taskid


def validate_key_prefix(key_prefix: object) -> str:
    if not isinstance(key_prefix, str) or not key_prefix:
        raise ValueError("COS_KEY_PREFIX must be a non-empty relative path")
    if key_prefix.startswith("/") or key_prefix.endswith("/"):
        raise ValueError("COS_KEY_PREFIX must be a relative path without leading or trailing '/'")
    if any(_KEY_PREFIX_SEGMENT_RE.fullmatch(segment) is None for segment in key_prefix.split("/")):
        raise ValueError("COS_KEY_PREFIX contains an invalid path segment")
    return key_prefix


def _authorization(
    config: CosUploadConfig, uri: str, content_type: str, start: int
) -> str:
    sign_time = f"{start};{start + _SIGNATURE_TTL}"
    host = f"{config.bucket}.cos.{config.region}.myqcloud.com"
    encode_value = lambda value: quote(value, safe="-_.~")
    canonical_headers = (
        f"content-type={encode_value(content_type)}&host={encode_value(host)}\n"
    )
    http_string = f"put\n{uri}\n\n{canonical_headers}"
    string_to_sign = (
        f"sha1\n{sign_time}\n{hashlib.sha1(http_string.encode('utf-8')).hexdigest()}\n"
    )
    sign_key = hmac.new(
        config.secret_key.encode("utf-8"), sign_time.encode("utf-8"), hashlib.sha1
    ).hexdigest()
    # COS V5 specifies the hexadecimal SignKey text itself as the next HMAC key.
    signature = hmac.new(
        sign_key.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1
    ).hexdigest()
    return (
        f"q-sign-algorithm=sha1&q-ak={quote(config.secret_id, safe='')}&"
        f"q-sign-time={sign_time}&q-key-time={sign_time}&"
        f"q-header-list=content-type;host&q-url-param-list=&q-signature={signature}"
    )


def upload_mapping_results(
    taskid: object,
    global_skus_bytes: bytes,
    viewer_bundle_bytes: bytes,
    config: CosUploadConfig,
    *,
    session=requests,
    now: Callable[[], int] = time.time,
) -> dict[str, str]:
    taskid = validate_taskid(taskid)
    key_prefix = validate_key_prefix(config.key_prefix)
    host = f"{config.bucket}.cos.{config.region}.myqcloud.com"
    results: dict[str, str] = {}
    for filename, content, content_type, result_key in (
        ("global_skus.json", global_skus_bytes, "application/json", "global_skus_url"),
        (
            "viewer_bundle.zip",
            viewer_bundle_bytes,
            "application/zip",
            "viewer_bundle_url",
        ),
    ):
        key = f"{key_prefix}/{taskid}/{filename}"
        encoded_key = quote(key, safe="/._-")
        uri = f"/{encoded_key}"
        url = f"https://{host}{uri}"
        headers = {
            "Content-Type": content_type,
            "Host": host,
            "Authorization": _authorization(config, uri, content_type, int(now())),
        }
        response = session.put(
            url,
            data=content,
            headers=headers,
            timeout=(5, 30),
            allow_redirects=False,
        )
        response.raise_for_status()
        if not 200 <= response.status_code < 300:
            raise requests.HTTPError(f"COS upload returned HTTP {response.status_code}")
        results[result_key] = url
    return results
