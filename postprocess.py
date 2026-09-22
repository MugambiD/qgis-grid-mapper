"""Detection post-processing: non-maximum suppression across overlapping tiles."""
import numpy as np


def nms(boxes, scores, iou_threshold=0.3, edge_flags=None):
    """Greedy NMS. boxes: (N, 4) array of xmin, ymin, xmax, ymax (any units).

    edge_flags: optional booleans marking boxes that touch their tile border
    (probably clipped). Those are dropped when half of them is covered by a
    kept box, so only the complete copy from the neighbouring tile survives.

    Returns indices of kept boxes, highest score first.
    """
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    edge = np.zeros(len(boxes), bool) if edge_flags is None else np.asarray(edge_flags, bool)
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0)
        # also suppress boxes mostly contained in the kept one (tile-edge fragments)
        contain = np.where(areas[rest] > 0, inter / areas[rest], 0)
        contain_limit = np.where(edge[rest], 0.5, 0.8)
        order = rest[(iou <= iou_threshold) & (contain <= contain_limit)]
    return keep


def edge_touching(xmin, ymin, xmax, ymax, w, h, margin=2):
    """True when a pixel box touches the tile border (likely a clipped object)."""
    return xmin <= margin or ymin <= margin or xmax >= w - margin or ymax >= h - margin
