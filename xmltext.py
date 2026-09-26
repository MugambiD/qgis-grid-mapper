"""Minimal XML text escaping.

Grid Mapper only ever *writes* small XML label files (Pascal VOC annotations and
world-file sidecars); it never parses XML from an untrusted source. This module
provides the one escaping helper that is needed, so the plugin does not import
``xml.sax`` at all.
"""

_REPLACEMENTS = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
    ('"', "&quot;"),
)


def escape(text):
    """Return *text* with the XML markup characters replaced by entities."""
    out = str("" if text is None else text)
    for char, entity in _REPLACEMENTS:
        out = out.replace(char, entity)
    return out
