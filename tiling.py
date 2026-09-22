"""Tiling and rendering helpers.

Any raster layer QGIS can draw (GeoTIFF, drone orthomosaic, WMS, XYZ such as
Google / Bing / Esri satellite) is rendered tile-by-tile into QImages at a
fixed ground resolution, exactly as the thesis chips were produced
(750 x 750 px screenshots of satellite basemaps in QGIS).
"""
import math

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsMapRendererCustomPainterJob,
    QgsMapSettings,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsUnitTypes,
)
from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice, QSize, Qt
from qgis.PyQt.QtGui import QColor, QImage, QPainter

WEB_MERCATOR = QgsCoordinateReferenceSystem("EPSG:3857")
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")


def choose_render_crs(layer_crs):
    """Use the layer CRS when it is projected, otherwise Web Mercator (XYZ tiles)."""
    if layer_crs is not None and layer_crs.isValid() and not layer_crs.isGeographic():
        return layer_crs
    return WEB_MERCATOR


def _metres_to_map_units(crs):
    try:  # QGIS >= 3.30
        from qgis.core import Qgis
        return QgsUnitTypes.fromUnitToUnitFactor(Qgis.DistanceUnit.Meters, crs.mapUnits())
    except (AttributeError, TypeError):  # QGIS < 3.30
        return QgsUnitTypes.fromUnitToUnitFactor(getattr(QgsUnitTypes, 'DistanceMeters'), crs.mapUnits())


def units_per_pixel(render_crs, gsd_m, center, transform_context):
    """Convert a ground sample distance (metres/pixel) to render-CRS units/pixel.

    Web Mercator metres are stretched by 1/cos(latitude), so the latitude of the
    area of interest is used to keep the true ground resolution constant.
    """
    if render_crs.authid() == "EPSG:3857":
        tr = QgsCoordinateTransform(render_crs, WGS84, transform_context)
        lat = tr.transform(center).y()
        return gsd_m / max(math.cos(math.radians(lat)), 1e-6)
    return gsd_m * _metres_to_map_units(render_crs)


def _grid_dims(extent, size, step):
    nx = 1 if extent.width() <= size else int(math.ceil((extent.width() - size) / step)) + 1
    ny = 1 if extent.height() <= size else int(math.ceil((extent.height() - size) / step)) + 1
    return nx, ny


def tile_grid(extent, tile_px, upp, overlap_frac, aoi_geom=None):
    """Yield (row, col, QgsRectangle) covering *extent* with square tiles.

    With an AOI only tiles touching it are produced; candidate cells are found
    per AOI part, so scattered zones across a whole country stay fast.
    """
    size = tile_px * upp
    step = size * (1.0 - overlap_frac)
    if step <= 0:
        raise ValueError("Overlap must be below 100 %")
    nx, ny = _grid_dims(extent, size, step)
    # centre the grid on the extent
    x0 = extent.center().x() - ((nx - 1) * step + size) / 2.0
    y1 = extent.center().y() + ((ny - 1) * step + size) / 2.0

    def rect_of(r, c):
        xmin = x0 + c * step
        ymax = y1 - r * step
        return QgsRectangle(xmin, ymax - size, xmin + size, ymax)

    if aoi_geom is None:
        for r in range(ny):
            for c in range(nx):
                yield r, c, rect_of(r, c)
        return

    cells = set()
    parts = aoi_geom.asGeometryCollection() if aoi_geom.isMultipart() else [aoi_geom]
    for part in parts:
        bb = part.boundingBox()
        c_lo = max(0, int(math.floor((bb.xMinimum() - x0 - size) / step)))
        c_hi = min(nx - 1, int(math.ceil((bb.xMaximum() - x0) / step)))
        r_lo = max(0, int(math.floor((y1 - bb.yMaximum() - size) / step)))
        r_hi = min(ny - 1, int(math.ceil((y1 - bb.yMinimum()) / step)))
        engine = QgsGeometry.createGeometryEngine(part.constGet())
        engine.prepareGeometry()
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                if (r, c) in cells:
                    continue
                cell_geom = QgsGeometry.fromRect(rect_of(r, c))  # keep alive while GEOS uses it
                if engine.intersects(cell_geom.constGet()):
                    cells.add((r, c))
    for r, c in sorted(cells):
        yield r, c, rect_of(r, c)


def count_tiles(extent, tile_px, upp, overlap_frac):
    size = tile_px * upp
    nx, ny = _grid_dims(extent, size, size * (1.0 - overlap_frac))
    return nx * ny


def square_rect(center, tile_px, upp):
    half = tile_px * upp / 2.0
    return QgsRectangle(center.x() - half, center.y() - half, center.x() + half, center.y() + half)


def render_tile(layers, crs, rect, tile_px, transform_context, background=QColor(0, 0, 0)):
    """Render *layers* into a tile_px x tile_px QImage covering *rect*.

    Safe to call from a Processing worker thread as long as *layers* are
    clones created on the main thread (see prepareAlgorithm).
    """
    ms = QgsMapSettings()
    ms.setLayers(layers)
    ms.setDestinationCrs(crs)
    ms.setOutputSize(QSize(tile_px, tile_px))
    ms.setOutputDpi(96)
    ms.setExtent(rect)
    ms.setBackgroundColor(background)
    ms.setTransformContext(transform_context)
    img = QImage(tile_px, tile_px, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(background)
    painter = QPainter(img)
    job = QgsMapRendererCustomPainterJob(ms, painter)
    job.start()
    job.waitForFinished()
    painter.end()
    return img, ms.visibleExtent()


def qimage_to_rgb(img, size=None):
    """QImage -> numpy uint8 array (H, W, 3). Optionally resize (w, h) first."""
    if size is not None and (img.width(), img.height()) != tuple(size):
        img = img.scaled(size[0], size[1], Qt.AspectRatioMode.IgnoreAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    ptr = img.constBits()
    ptr.setsize(h * bpl)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, bpl)[:, : w * 3]
    return arr.reshape(h, w, 3).copy()


def blank_fraction(rgb, threshold=8):
    """Fraction of (near-)black pixels - used to skip tiles with no imagery."""
    return float((rgb.max(axis=2) <= threshold).mean())


def qimage_to_bytes(img, fmt="JPG", quality=90):
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.convertToFormat(QImage.Format.Format_RGB888).save(buf, fmt, quality)
    buf.close()
    return bytes(ba)


def pixel_to_map(rect, tile_px_w, tile_px_h, px, py):
    """Pixel (column, row) inside a rendered tile -> map coordinate."""
    x = rect.xMinimum() + px * rect.width() / tile_px_w
    y = rect.yMaximum() - py * rect.height() / tile_px_h
    return QgsPointXY(x, y)


def world_file_lines(rect, width_px, height_px):
    """ESRI world file (.jgw/.pgw) contents, referencing pixel centres."""
    a = rect.width() / width_px
    e = -rect.height() / height_px
    c = rect.xMinimum() + a / 2.0
    f = rect.yMaximum() + e / 2.0
    return [f"{a:.12f}", "0.0", "0.0", f"{e:.12f}", f"{c:.12f}", f"{f:.12f}"]


def to_project_transform(src_crs, dst_crs, context=None):
    return QgsCoordinateTransform(src_crs, dst_crs, context or QgsProject.instance().transformContext())
