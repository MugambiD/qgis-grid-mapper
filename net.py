"""HTTP helpers using QGIS' network stack (proxy, SSL and auth settings)."""
import json
import urllib.parse

USER_AGENT = "QGIS-GridMapper/1.5.0 (substation and grid mapping plugin)"


class HttpError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def request(url, data=None, headers=None, timeout_ms=180000):
    """GET (data is None) or POST through QGIS. Returns ``(status, bytes)``."""
    headers = dict(headers or {})
    headers.setdefault("User-Agent", USER_AGENT)
    try:
        from qgis.core import QgsBlockingNetworkRequest
        from qgis.PyQt.QtCore import QByteArray, QUrl
        from qgis.PyQt.QtNetwork import QNetworkRequest
    except ImportError as exc:  # pragma: no cover - runtime helper belongs inside QGIS
        raise HttpError(0, "QGIS network APIs are unavailable") from exc

    req = QNetworkRequest(QUrl(url))
    for key, value in headers.items():
        req.setRawHeader(QByteArray(key.encode()), QByteArray(str(value).encode()))
    try:
        req.setTransferTimeout(timeout_ms)
    except AttributeError:
        pass

    blocking = QgsBlockingNetworkRequest()
    blocking.get(req, True) if data is None else blocking.post(req, QByteArray(data))
    reply = blocking.reply()
    try:  # Qt6 / QGIS 4
        status_attr = QNetworkRequest.Attribute.HttpStatusCodeAttribute
    except AttributeError:  # Qt5 / QGIS 3
        status_attr = QNetworkRequest.HttpStatusCodeAttribute
    status = reply.attribute(status_attr)
    content = bytes(reply.content())
    if status:
        return int(status), content
    message = blocking.errorMessage() or reply.errorString() or "no response"
    raise HttpError(0, message)


def get_json(url, params=None, timeout_ms=180000):
    if params:
        url = f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode(params)}"
    status, content = request(url, timeout_ms=timeout_ms)
    return _decode(status, content, url)


def post_form_json(url, form, timeout_ms=600000):
    body = urllib.parse.urlencode(form).encode()
    status, content = request(url, body, {"Content-Type": "application/x-www-form-urlencoded"}, timeout_ms)
    return _decode(status, content, url)


def _decode(status, content, url):
    host = url.split("/")[2] if "//" in url else url
    if status >= 400:
        text = content[:2000].decode("utf-8", "ignore")
        # Overpass returns HTML error pages - keep only the readable remark.
        import re
        remark = re.findall(r"<strong[^>]*>(.*?)</strong>\s*(.*?)</p>", text, re.S)
        detail = " ".join(" ".join(r) for r in remark)[:200] if remark else re.sub(r"<[^>]+>", " ", text)
        detail = " ".join(detail.split())[:200]
        raise HttpError(status, f"HTTP {status} from {host}: {detail}")
    try:
        return json.loads(content.decode("utf-8"))
    except ValueError as exc:
        raise HttpError(status, f"Unexpected response from {host}: {content[:200]!r}") from exc
