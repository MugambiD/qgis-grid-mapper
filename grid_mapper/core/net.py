"""HTTP helpers that go through QGIS' network manager (proxy, SSL, auth settings)."""
import json
import urllib.parse

USER_AGENT = "QGIS-GridMapper/1.2 (substation & grid mapping plugin)"


def request(url, data=None, headers=None, timeout_ms=180000):
    """GET (data is None) or POST. Returns (status, bytes)."""
    headers = dict(headers or {})
    headers.setdefault("User-Agent", USER_AGENT)
    try:
        from qgis.core import QgsBlockingNetworkRequest
        from qgis.PyQt.QtCore import QByteArray, QUrl
        from qgis.PyQt.QtNetwork import QNetworkRequest
    except ImportError:  # outside QGIS (unit tests)
        import urllib.request
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_ms / 1000) as resp:  # noqa: S310
            return resp.status, resp.read()

    req = QNetworkRequest(QUrl(url))
    for k, v in headers.items():
        req.setRawHeader(QByteArray(k.encode()), QByteArray(str(v).encode()))
    try:
        req.setTransferTimeout(timeout_ms)
    except AttributeError:
        pass
    blocking = QgsBlockingNetworkRequest()
    err = blocking.get(req, True) if data is None else blocking.post(req, QByteArray(data))
    reply = blocking.reply()
    status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
    content = bytes(reply.content())
    if not content and not status:
        raise RuntimeError(blocking.errorMessage() or f"Network error {err} for {url}")
    return int(status or 0), content


def get_json(url, params=None, timeout_ms=180000):
    if params:
        url = f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode(params)}"
    status, content = request(url, timeout_ms=timeout_ms)
    return _decode(status, content, url)


def post_form_json(url, form, timeout_ms=900000):
    body = urllib.parse.urlencode(form).encode()
    status, content = request(url, body, {"Content-Type": "application/x-www-form-urlencoded"}, timeout_ms)
    return _decode(status, content, url)


def _decode(status, content, url):
    if status >= 400:
        raise RuntimeError(f"HTTP {status} from {url.split('?')[0]}: {content[:300].decode('utf-8', 'ignore')}")
    try:
        return json.loads(content.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"Unexpected response from {url.split('?')[0]}: {content[:300]!r}") from exc
