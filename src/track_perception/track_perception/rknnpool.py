from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import cloudpickle
import sys


def initRKNN(rknnModel="yolov.rknn", id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn_lite


def initRKNNs(rknnModel="yolo.rknn", TPEs=1, core_ids=None):
    sys.modules['pickle'] = cloudpickle
    rknn_list = []
    if core_ids:
        core_sequence = [int(core_id) for core_id in core_ids]
    else:
        core_sequence = [0, 1, 2]

    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, core_sequence[i % len(core_sequence)]))
    return rknn_list


class rknnPoolExecutor():
    def __init__(self, rknnModel, TPEs, func, core_ids=None):
        self.TPEs = TPEs
        self.queue = Queue()
        self.core_ids = [int(core_id) for core_id in core_ids] if core_ids else None
        self.rknnPool = initRKNNs(rknnModel, TPEs, self.core_ids)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        # self.pool = ProcessPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(self.pool.submit(
            self.func, self.rknnPool[self.num % self.TPEs], frame))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
