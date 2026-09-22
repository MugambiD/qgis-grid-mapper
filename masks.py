"""Raster mask -> polygon helpers (GDAL, available in every QGIS install)."""
import numpy as np


def mask_to_polygons(mask, keep_values, scale_x, scale_y, min_pixels=20):
    """Vectorise the pixels of *mask* whose value is in *keep_values*.

    Returns a list of outer rings in pixel space, scaled by (scale_x, scale_y).
    """
    import contextlib

    from osgeo import gdal, ogr

    # scoped exception mode: silences GDAL's FutureWarning without changing QGIS' global setting
    ctx = gdal.ExceptionMgr(useExceptions=True) if hasattr(gdal, "ExceptionMgr") else contextlib.nullcontext()
    with ctx:
        return _polygonize(gdal, ogr, mask, keep_values, scale_x, scale_y, min_pixels)


def _polygonize(gdal, ogr, mask, keep_values, scale_x, scale_y, min_pixels):
    binary = np.isin(mask, list(keep_values)).astype(np.uint8)
    if binary.sum() < min_pixels:
        return []
    h, w = binary.shape
    ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((0, 1, 0, 0, 0, 1))  # map coords == pixel coords
    band = ds.GetRasterBand(1)
    band.WriteArray(binary)
    drv = ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")  # GDAL >= 3.11 / older
    vds = drv.CreateDataSource("mask")
    lyr = vds.CreateLayer("poly", geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0, [], callback=None)
    rings = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        if geom is None or feat.GetField("v") != 1 or geom.GetArea() < min_pixels:
            continue
        geom = geom.SimplifyPreserveTopology(1.0)
        outer = geom.GetGeometryRef(0)
        rings.append([(outer.GetX(i) * scale_x, outer.GetY(i) * scale_y)
                      for i in range(outer.GetPointCount())])
    return rings


def largest_ring(rings):
    def area(r):
        return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(r, r[1:] + r[:1]))) / 2.0
    return max(rings, key=area) if rings else None
