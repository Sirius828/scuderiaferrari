"""
RKNN 线程池执行器
基于官方 setupUI 的 rknnpool.py 实现
支持多 NPU 核心并行推理，提升吞吐量
"""

from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor
import cloudpickle
import sys


def init_rknn(rknn_model, core_id=0):
    """
    初始化单个 RKNN 实例并绑定到指定 NPU 核心
    
    Args:
        rknn_model: RKNN 模型路径
        core_id: NPU 核心 ID (0, 1, 2 或 -1 表示全部)
    
    Returns:
        RKNNLite 实例
    """
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print(f"❌ Load RKNN model failed: {rknn_model}")
        exit(ret)
    
    # 绑定到指定 NPU 核心
    if core_id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif core_id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif core_id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif core_id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    
    if ret != 0:
        print("❌ Init runtime environment failed")
        exit(ret)
    
    print(f"✅ RKNN instance initialized on NPU Core {core_id}: {rknn_model}")
    return rknn_lite


def init_rknn_pool(rknn_model, tpes=3, core_ids=None):
    """
    初始化多个 RKNN 实例（每个绑定到不同的 NPU 核心）
    
    Args:
        rknn_model: RKNN 模型路径
        tpes: 线程池执行器数量（通常等于 NPU 核心数）
        core_ids: NPU 核心 ID 列表，例如 [0, 1]；None 表示按 0/1/2 轮询
    
    Returns:
        RKNN 实例列表
    """
    # 使用 cloudpickle 替代标准 pickle，支持更复杂的对象序列化
    sys.modules['pickle'] = cloudpickle
    
    rknn_list = []
    if core_ids:
        core_sequence = [int(core_id) for core_id in core_ids]
    else:
        core_sequence = [0, 1, 2]

    for i in range(tpes):
        # 按指定核心列表轮询；未指定时默认轮询 0/1/2。
        core_id = core_sequence[i % len(core_sequence)]
        rknn_list.append(init_rknn(rknn_model, core_id))
    
    return rknn_list


class RKNNPoolExecutor:
    """
    RKNN 线程池执行器
    
    使用多个 RKNN 实例并行处理推理请求，显著提升吞吐量。
    工作流程：
    1. put() 提交图像到线程池（非阻塞）
    2. 线程池中的工作线程调用推理函数
    3. get() 从队列获取结果（阻塞等待）
    """
    
    def __init__(self, rknn_model, tpes=3, func=None, core_ids=None):
        """
        初始化 RKNN 线程池执行器
        
        Args:
            rknn_model: RKNN 模型路径
            tpes: 线程池大小（建议设置为 NPU 核心数，通常为 3）
            func: 推理回调函数，签名: func(rknn_instance, input_data) -> result
            core_ids: NPU 核心 ID 列表，例如 [0, 1]；None 表示按 0/1/2 轮询
        """
        self.tpes = tpes
        self.queue = Queue()
        self.core_ids = [int(core_id) for core_id in core_ids] if core_ids else None
        
        # 创建多个 RKNN 实例（每个绑定到不同的 NPU 核心）
        self.rknn_pool = init_rknn_pool(rknn_model, tpes, self.core_ids)
        
        # 创建线程池
        self.pool = ThreadPoolExecutor(max_workers=tpes)
        
        # 推理回调函数
        self.func = func
        
        # 计数器，用于轮询分配 RKNN 实例
        self.num = 0
    
    def put(self, frame):
        """
        提交一帧图像到线程池进行推理（非阻塞）
        
        Args:
            frame: 输入图像（numpy array）
        """
        # 轮询选择一个 RKNN 实例
        rknn_instance = self.rknn_pool[self.num % self.tpes]
        
        # 提交到线程池，立即返回 Future 对象
        future = self.pool.submit(self.func, rknn_instance, frame)
        
        # 将 Future 放入队列
        self.queue.put(future)
        
        # 更新计数器
        self.num += 1
    
    def get(self):
        """
        从队列获取推理结果（阻塞等待）
        
        Returns:
            tuple: (result, flag)
                - result: 推理结果
                - flag: 是否成功（True/False）
        """
        if self.queue.empty():
            return None, False
        
        # 从队列获取 Future 对象
        future = self.queue.get()
        
        # 等待并获取结果（阻塞）
        try:
            result = future.result()
            return result, True
        except Exception as e:
            print(f"❌ RKNN inference error: {e}")
            return None, False
    
    def release(self):
        """释放资源"""
        # 关闭线程池
        self.pool.shutdown(wait=True)
        
        # 释放所有 RKNN 实例
        for rknn_lite in self.rknn_pool:
            rknn_lite.release()
        
        print("🔒 RKNN pool resources released")
