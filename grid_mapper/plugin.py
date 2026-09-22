import os

from qgis.core import QgsApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QMenu

try:  # Qt6 / QGIS 4
    from qgis.PyQt.QtGui import QAction
except ImportError:  # Qt5 / QGIS 3
    from qgis.PyQt.QtWidgets import QAction

from .provider import GridMapperProvider, PROVIDER_ID

PLUGIN_DIR = os.path.dirname(__file__)
MENU_TITLE = "&Grid Mapper"

ACTIONS = [
    ("map_grid", "Map the grid (type a place)…", True),
    ("country_scan", "Scan an entire country (resumable)…", True),
    ("detect_tflite", "Detect substations (local model: ONNX / TFLite)…", True),
    ("prioritise_review", "Prioritise detections for review…", True),
    ("collect_feedback", "Capture reviewed feedback for learning…", True),
    ("export_feedback_dataset", "Export continual-learning dataset snapshot…", False),
    ("register_candidate_model", "Register / safely promote a candidate model…", False),
    ("detect_roboflow", "Detect substations (Roboflow API)…", False),
    ("extract_chips", "Extract training chips around substations…", False),
    ("validate_detections", "Validate detections against reference…", False),
]

class GridMapperPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.actions = []
        self.menu = None
        self.toolbar = None

    def initProcessing(self):  # noqa: N802
        self.provider = GridMapperProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):  # noqa: N802
        self.initProcessing()
        icon = QIcon(os.path.join(PLUGIN_DIR, "icons", "icon.png"))
        self.menu = QMenu(MENU_TITLE.replace("&", ""), self.iface.mainWindow())
        self.menu.setIcon(icon)
        self.toolbar = self.iface.addToolBar("Grid Mapper")
        self.toolbar.setObjectName("GridMapperToolbar")
        for alg_id, label, on_toolbar in ACTIONS:
            action = QAction(icon, label, self.iface.mainWindow())
            action.triggered.connect(lambda _checked=False, a=alg_id: self.open_algorithm(a))
            self.menu.addAction(action)
            if on_toolbar:
                self.toolbar.addAction(action)
            self.actions.append(action)
        self.iface.pluginMenu().addMenu(self.menu)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(MENU_TITLE, action)
        if self.menu is not None:
            self.iface.pluginMenu().removeAction(self.menu.menuAction())
            self.menu.deleteLater()
        if self.toolbar is not None:
            self.toolbar.deleteLater()
        self.actions = []
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None

    @staticmethod
    def open_algorithm(alg_id):
        import processing  # available once QGIS Processing is loaded
        processing.execAlgorithmDialog(f"{PROVIDER_ID}:{alg_id}")
