"""Grid Mapper - Substation Detector QGIS plugin."""


def classFactory(iface):  # noqa: N802 (QGIS API name)
    from .plugin import GridMapperPlugin
    return GridMapperPlugin(iface)
