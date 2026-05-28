from __future__ import annotations

import csv
from functools import lru_cache
import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, Optional

from .config import HttpConfig


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _build_url(url: str, params: Optional[Dict[str, object]]) -> str:
    if not params:
        return url
    encoded = urllib.parse.urlencode(params, doseq=True)
    return f"{url}?{encoded}" if encoded else url


def _request(
    url: str,
    config: HttpConfig,
    *,
    data: bytes | None = None,
    headers: Optional[Dict[str, str]] = None,
    method: str | None = None,
) -> urllib.request.Request:
    request_headers = {"User-Agent": config.user_agent}
    if headers:
        request_headers.update(headers)
    return urllib.request.Request(url, data=data, headers=request_headers, method=method)


def _ca_bundle_path(config: HttpConfig) -> str | None:
    for value in (
        config.ca_bundle_path,
        os.getenv("HERG_CA_BUNDLE"),
        os.getenv("SSL_CERT_FILE"),
        os.getenv("REQUESTS_CA_BUNDLE"),
    ):
        if value and value.strip():
            return value.strip()
    return None


@lru_cache(maxsize=8)
def _ssl_context(ca_bundle_path: str | None) -> ssl.SSLContext | None:
    if not ca_bundle_path:
        return None
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=ca_bundle_path)
    return context


def _urlopen(req: urllib.request.Request, config: HttpConfig):
    context = _ssl_context(_ca_bundle_path(config))
    if context is None:
        return urllib.request.urlopen(req, timeout=config.request_timeout_seconds)
    return urllib.request.urlopen(req, timeout=config.request_timeout_seconds, context=context)


def get_json(
    url: str,
    params: Optional[Dict[str, object]],
    config: HttpConfig,
    label: str = "API",
) -> Dict:
    full_url = _build_url(url, params)

    for attempt in range(config.http_retries + 1):
        try:
            req = _request(full_url, config)
            with _urlopen(req, config) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if attempt < config.http_retries and exc.code in RETRYABLE_STATUS:
                time.sleep(2**attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < config.http_retries:
                time.sleep(2**attempt)
                continue
            raise
        except http.client.RemoteDisconnected:
            if attempt < config.http_retries:
                time.sleep(2**attempt)
                continue
            raise

    return {}


def post_json(
    url: str,
    payload: dict,
    config: HttpConfig,
    label: str = "API",
) -> Dict:
    encoded_payload = json.dumps(payload).encode("utf-8")

    for attempt in range(config.http_retries + 1):
        try:
            req = _request(
                url,
                config,
                data=encoded_payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with _urlopen(req, config) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if attempt < config.http_retries and exc.code in RETRYABLE_STATUS:
                time.sleep(2**attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < config.http_retries:
                time.sleep(2**attempt)
                continue
            raise
        except http.client.RemoteDisconnected:
            if attempt < config.http_retries:
                time.sleep(2**attempt)
                continue
            raise

    return {}


def get_csv_rows(url: str, config: HttpConfig, label: str = "API") -> Iterable[dict]:
    for attempt in range(config.http_retries + 1):
        try:
            req = _request(url, config)
            with _urlopen(req, config) as response:
                reader = csv.DictReader((line.decode("utf-8", "ignore") for line in response))
                for row in reader:
                    yield row
            return
        except urllib.error.HTTPError as exc:
            if attempt < config.http_retries and exc.code in RETRYABLE_STATUS:
                time.sleep(2**attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < config.http_retries:
                time.sleep(2**attempt)
                continue
            raise
        except http.client.RemoteDisconnected:
            if attempt < config.http_retries:
                time.sleep(2**attempt)
                continue
            raise
