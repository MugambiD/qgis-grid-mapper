import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algorithms.active_learning import PrioritiseReview
from .algorithms.collect_feedback import CollectTrainingFeedback
from .algorithms.country_scan import CountryScan
from .algorithms.detect_tflite import DetectSubstationsTFLite
from .algorithms.detect_roboflow import DetectSubstationsRoboflow
from .algorithms.export_feedback import ExportFeedbackDataset
from .algorithms.extract_chips import ExtractTrainingChips
from .algorithms.manage_models import RegisterCandidateModel
from .algorithms.map_grid import MapGrid
from .algorithms.validate import ValidateDetections

PROVIDER_ID = "gridmapper"


class GridMapperProvider(QgsProcessingProvider):
    def loadAlgorithms(self):  # noqa: N802
        for alg in (
            MapGrid(),
            CountryScan(),
            DetectSubstationsTFLite(),
            DetectSubstationsRoboflow(),
            PrioritiseReview(),
            CollectTrainingFeedback(),
            ExportFeedbackDataset(),
            RegisterCandidateModel(),
            ExtractTrainingChips(),
            ValidateDetections(),
        ):
            self.addAlgorithm(alg)

    def id(self):
        return PROVIDER_ID

    def name(self):
        return "Grid Mapper"

    def longName(self):  # noqa: N802
        return "Grid Mapper - substation detection and continual learning"

    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), "icons", "icon.png"))
