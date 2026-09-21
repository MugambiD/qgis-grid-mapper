"""QGIS 3.22 - 4.x (Qt5 / Qt6) compatibility shims.

QGIS 4 (PyQt6) removed the old unscoped enum aliases and QVariant.Type, so
every enum is resolved here: newest scoped name first, legacy name second.
"""
from qgis.core import (
    Qgis,
    QgsFeatureSink,
    QgsField,
    QgsProcessing,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsWkbTypes,
)


def _pick(*candidates):
    for owner, path in candidates:
        obj = owner
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"None of {[p for _, p in candidates]} exist in this QGIS version")


NUM_DOUBLE = _pick((Qgis, "ProcessingNumberParameterType.Double"), (QgsProcessingParameterNumber, "Double"))
NUM_INT = _pick((Qgis, "ProcessingNumberParameterType.Integer"), (QgsProcessingParameterNumber, "Integer"))

SRC_POLYGON = _pick((Qgis, "ProcessingSourceType.VectorPolygon"), (QgsProcessing, "TypeVectorPolygon"))
SRC_POINT = _pick((Qgis, "ProcessingSourceType.VectorPoint"), (QgsProcessing, "TypeVectorPoint"))
SRC_ANY = _pick((Qgis, "ProcessingSourceType.VectorAnyGeometry"), (QgsProcessing, "TypeVectorAnyGeometry"))

FILE_BEHAVIOR = _pick((Qgis, "ProcessingFileParameterBehavior.File"), (QgsProcessingParameterFile, "File"))
FIELD_NUMERIC = _pick((Qgis, "ProcessingFieldParameterDataType.Numeric"),
                      (QgsProcessingParameterField, "Numeric"))

WKB_POLYGON = _pick((Qgis, "WkbType.Polygon"), (QgsWkbTypes, "Polygon"))
WKB_POINT = _pick((Qgis, "WkbType.Point"), (QgsWkbTypes, "Point"))

FAST_INSERT = _pick((QgsFeatureSink, "Flag.FastInsert"), (QgsFeatureSink, "FastInsert"))


def _field_types():
    if Qgis.QGIS_VERSION_INT >= 33800:  # QgsField takes QMetaType.Type from 3.38 (required in QGIS 4)
        from qgis.PyQt.QtCore import QMetaType
        t = QMetaType.Type
        return {"int": t.Int, "long": t.LongLong, "double": t.Double, "string": t.QString}
    from qgis.PyQt.QtCore import QVariant
    return {"int": QVariant.Int, "long": QVariant.LongLong, "double": QVariant.Double,
            "string": QVariant.String}


_TYPES = _field_types()


def make_field(name, kind, length=0, prec=0):
    return QgsField(name, _TYPES[kind], len=length, prec=prec)

WKB_LINESTRING = _pick((Qgis, "WkbType.LineString"), (QgsWkbTypes, "LineString"))


def _flag_advanced():
    from qgis.core import QgsProcessingParameterDefinition
    return _pick((Qgis, "ProcessingParameterFlag.Advanced"), (QgsProcessingParameterDefinition, "FlagAdvanced"))


FLAG_ADVANCED = _flag_advanced()


def advanced(param):
    """Mark a Processing parameter as 'Advanced' (collapsed in the dialog)."""
    param.setFlags(param.flags() | FLAG_ADVANCED)
    return param
