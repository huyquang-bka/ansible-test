import cv2
from .ocsort import OCSort
from .sort import Sort
import numpy as np
from ctypes import *
import numpy.ctypeslib as npct


class Detection:
    def __init__(self):
        super().__init__()
        self.weight_path = "resources/Weights/person_24_08_320.engine"
        self.max_bbox = 1000
        self.conf_thres = 0.5
        self.dll_path = "resources/Weights/libyolov5_person.so",
        self.classes = None

    def load_model(self):
        self.detector = CDLL(self.dll_path)
        self.detector.Detect.argtypes = [c_void_p, c_int, c_int, POINTER(c_ubyte),
                                         npct.ndpointer(dtype=np.float32, ndim=2, shape=(self.max_bbox, 6),
                                                        flags="C_CONTIGUOUS")]
        self.detector.Init.restype = c_void_p

        self.c_point = self.detector.Init(
            bytes(self.weight_path, encoding='utf-8'))
        self.classes = [
            i for i in 1000] if self.classes is None else self.classes

    def xywh2xyxy(self, boxes):
        bboxes = []
        for bbox in boxes:
            x, y, w, h, cls, conf = bbox
            if conf < self.conf_thres or cls not in self.classes:
                continue
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            x1, y1, x2, y2, cls = list(
                map(lambda x: max(0, int(x)), [x1, y1, x2, y2, cls]))
            bboxes.append([x1, y1, x2, y2, cls, conf])
        return bboxes

    def detect(self, img):
        rows, cols = img.shape[:2]
        res_arr = np.zeros((self.max_bbox, 6), dtype=np.float32)
        self.detector.Detect(self.c_point, c_int(rows), c_int(
            cols), img.ctypes.data_as(POINTER(c_ubyte)), res_arr)
        bboxes = res_arr[~(res_arr == 0).all(1)]
        return self.xywh2xyxy(bboxes)


class Tracking(Detection):
    def __init__(self, max_age=70, min_hits=3, iou_threshold=0.3):
        super().__init__()
        self._tracker = Sort(
            max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def track(self, image):
        track_dict = {}
        bboxes = self.detect(image)
        dets_to_sort = np.empty((0, 6))
        for x1, y1, x2, y2, cls, conf in bboxes:
            dets_to_sort = np.vstack(
                (dets_to_sort, np.array([x1, y1, x2, y2, conf, cls])))

        tracked_det = self._tracker.update(dets_to_sort)
        if len(tracked_det) > 0:
            bbox_xyxy = tracked_det[:, :4]
            indentities = tracked_det[:, 8]
            categories = tracked_det[:, 4]
            for i in range(len(bbox_xyxy)):
                x1, y1, x2, y2 = list(
                    map(lambda x: max(0, int(x)), bbox_xyxy[i]))
                id = int(indentities[i])
                track_dict[id] = [x1, y1, x2, y2, categories[i]]
        return track_dict


class OCTracking(Detection):
    def __init__(self, det_thresh=0.4, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        super().__init__()
        self.tracker = OCSort(det_thresh=det_thresh, max_age=max_age, min_hits=min_hits,
                              iou_threshold=iou_threshold, delta_t=delta_t, asso_func=asso_func, inertia=inertia, use_byte=use_byte)

    def track(self, image):
        track_dict = {}
        bboxes = self.detect(image)
        dets_to_sort = np.empty((0, 6))
        for x1, y1, x2, y2, cls, conf in bboxes:
            dets_to_sort = np.vstack(
                (dets_to_sort, np.array([x1, y1, x2, y2, conf, cls])))

        tracked_det = self.tracker.update(dets_to_sort)
        if len(tracked_det) > 0:
            bbox_xyxy = tracked_det[:, :4]
            indentities = tracked_det[:, 4]
            categories = tracked_det[:, 5]
            for i in range(len(bbox_xyxy)):
                x1, y1, x2, y2 = list(
                    map(lambda x: max(0, int(x)), bbox_xyxy[i]))
                id = int(indentities[i])
                track_dict[id] = [x1, y1, x2, y2, categories[i]]
        return track_dict


if __name__ == "__main__":
    detector = Detection()
    detector.weight_path = "resources/Weights/person_24_08_320.engine"  # tensorrt engine file
    detector.dll_path = "resources/Weights/libyolov5_person.so"  # .so file
    detector.classes = None  # or [0, 1, 2]
    detector.conf_thres = 0.5  # confidence threshold

    detector.load_model()

    path = "test.jpg"
    img = cv2.imread(path)
    print(detector.detect(img))
