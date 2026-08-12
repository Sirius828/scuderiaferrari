import cv2
import numpy as np
import onnxruntime as ort
import os
import sys
import requests
import json
import time

# 自动适配路径，不受终端工作目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from utils.db_postprocess import DBPostProcess

# ====================== 百度千帆API配置 ======================
QIANFAN_API_KEY = os.environ.get("QIANFAN_API_KEY", "")
LLM_MODEL = "ernie-3.5-8k"
API_URL = "https://qianfan.baidubce.com/v2/chat/completions"

# 全局复用HTTP长连接
api_session = requests.Session()

def get_dir_code(ocr_text):
    if not QIANFAN_API_KEY:
        raise RuntimeError("QIANFAN_API_KEY is not set")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QIANFAN_API_KEY}"
    }

    system_rule = """
仅输出0或1。
判定逻辑（按优先级执行）：
1. 先找出被明确否定、禁止、走不通的方向，直接排除；按剩余可通行的方向判定结果。
2. 「…才怪」为反语句式，按字面意思取反理解。
3. 直行/左转输出0，右转输出1。「抄近道」固定代表右转，输出1。
4. 无法明确判断默认输出0。
示例：“右道崎岖，但直道走不了” → 直道被否定排除，只剩右道可走 → 输出1
"""
    
    post_data = {
        "model": LLM_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_rule},
            {"role": "user", "content": ocr_text}
        ]
    }

    try:
        t_api_start = time.time()
        res = api_session.post(API_URL, headers=headers, json=post_data, timeout=6)
        t_api_end = time.time()
        print(f"\n【大模型API调用耗时】：{t_api_end - t_api_start:.3f} 秒")

        res_data = res.json()
        print("\n【百度接口完整返回数据】：", res_data)

        if "error" in res_data:
            raise Exception(f"平台错误：{res_data['error']}")

        out = res_data["choices"][0]["message"]["content"].strip()
        print(f"【大模型原始返回内容】：{out}")
        return 1 if "1" in out else 0

    except Exception as e:
        print(f"API调用异常：{e}")
        return 0
# ==============================================================================

# 路径完全匹配你的文件夹结构
DET_MODEL_PATH = os.path.join(BASE_DIR, "../model/ppocrv4_det.onnx")
REC_MODEL_PATH = os.path.join(BASE_DIR, "../model/ppocrv4_rec.onnx")
DICT_PATH = os.path.join(BASE_DIR, "../model/ppocr_keys_v1.txt")
TEST_IMG_PATH = os.path.join(BASE_DIR, "../model/test16.jpg")
RESULT_IMG_PATH = os.path.join(BASE_DIR, "../result.jpg")

# 加载字符合典
def load_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8-sig') as f:
        chars = [line.rstrip('\n').rstrip('\r') for line in f.readlines()]
    return [c for c in chars if c != '']

# 检测模型预处理：固定480x480
def det_preprocess(img):
    target_size = 480
    src_h, src_w = img.shape[:2]

    img_resized = cv2.resize(img, (target_size, target_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std

    img_chw = img_norm.transpose((2, 0, 1))
    input_data = np.expand_dims(img_chw, axis=0).astype(np.float32)
    return input_data, src_h, src_w, target_size, target_size

# 识别模型预处理：固定48x320
def rec_preprocess(img):
    target_h = 48
    target_w = 320
    img = cv2.resize(img, (target_w, target_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose((2, 0, 1))
    return np.expand_dims(img, axis=0).astype(np.float32)

# CTC解码
def rec_decode(preds, chars):
    pred = preds[0]
    argmax_idx = np.argmax(pred, axis=1)
    result = []
    prev_idx = -1
    blank_idx = 0
    for idx in argmax_idx:
        if idx != prev_idx and idx != blank_idx and idx <= len(chars):
            result.append(chars[idx - 1])
        prev_idx = idx
    return ''.join(result)

# 裁剪文本框
def crop_text_box(img, box):
    x_min = int(max(0, np.min(box[:, 0])))
    x_max = int(min(img.shape[1], np.max(box[:, 0])))
    y_min = int(max(0, np.min(box[:, 1])))
    y_max = int(min(img.shape[0], np.max(box[:, 1])))
    return img[y_min:y_max, x_min:x_max]

# 获取文本框平均Y坐标，用于从上到下排序
def get_box_avg_y(box_points):
    y_coords = [point[1] for point in box_points]
    return sum(y_coords) / len(y_coords)

if __name__ == "__main__":
    t_total_start = time.time()

    # ========== 1. 模型初始化阶段 ==========
    t_model_start = time.time()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4

    det_session = ort.InferenceSession(DET_MODEL_PATH, sess_options=so, providers=['CPUExecutionProvider'])
    det_input_name = det_session.get_inputs()[0].name
    rec_session = ort.InferenceSession(REC_MODEL_PATH, sess_options=so, providers=['CPUExecutionProvider'])
    rec_input_name = rec_session.get_inputs()[0].name

    db_post = DBPostProcess(
        thresh=0.3,
        box_thresh=0.6,
        max_candidates=1000,
        unclip_ratio=1.5
    )
    chars = load_dict(DICT_PATH)
    t_model_end = time.time()
    print(f"【模型加载与初始化耗时】：{t_model_end - t_model_start:.3f} 秒")

    # ========== 2. 图片读取 ==========
    t_img_start = time.time()
    img = cv2.imread(TEST_IMG_PATH)
    if img is None:
        print("图片读取失败，请检查路径")
        exit()
    img_visual = img.copy()
    t_img_end = time.time()
    print(f"【图片读取耗时】：{t_img_end - t_img_start:.3f} 秒")

    # ========== 3. 文本检测（预处理+推理+后处理） ==========
    t_det_start = time.time()
    det_input, src_h, src_w, new_h, new_w = det_preprocess(img)
    det_output = det_session.run(None, {det_input_name: det_input})
    pred_map = det_output[0]

    shape_list = [[src_h, src_w, new_h, new_w]]
    boxes_list = db_post({'maps': pred_map}, shape_list)
    result_dict = boxes_list[0]
    boxes = result_dict['points']
    boxes = sorted(boxes, key=get_box_avg_y)
    t_det_end = time.time()
    print(f"【文本检测全流程耗时】：{t_det_end - t_det_start:.3f} 秒（检测到 {len(boxes)} 个文本框）")

    # ========== 4. 文本识别 ==========
    t_rec_start = time.time()
    all_rec_text = []
    for i, box in enumerate(boxes):
        box = np.array(box).astype(np.int32)
        text_img = crop_text_box(img, box)
        if text_img.size == 0 or text_img.shape[0] < 2 or text_img.shape[1] < 2:
            continue

        rec_input = rec_preprocess(text_img)
        rec_output = rec_session.run(None, {rec_input_name: rec_input})
        text = rec_decode(rec_output[0], chars)
        all_rec_text.append(text)

        cv2.polylines(img_visual, [box], True, (0, 0, 255), 2)
        cv2.putText(img_visual, text, (box[0][0], max(10, box[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(f"第{i+1}行（图片从上至下顺序）：{text}")
    t_rec_end = time.time()
    print(f"【文本识别全流程耗时】：{t_rec_end - t_rec_start:.3f} 秒")

    # ========== 5. 结果图保存 ==========
    t_save_start = time.time()
    cv2.imwrite(RESULT_IMG_PATH, img_visual)
    t_save_end = time.time()
    print(f"【结果图保存耗时】：{t_save_end - t_save_start:.3f} 秒")
    print(f"\n带标注的结果图已保存：{RESULT_IMG_PATH}")

    # ========== 6. 大模型方向判断 ==========
    total_text = "\n".join(all_rec_text)
    print(f"\n【送入大模型的全部OCR文本（从上到下顺序）】：\n{total_text}")
    final_output = get_dir_code(total_text)
    print(f"\n【最终程序输出指令】：{final_output}")

    # ========== 总耗时统计 ==========
    t_total_end = time.time()
    print("\n" + "="*50)
    print(f"【程序总运行耗时】：{t_total_end - t_total_start:.3f} 秒")
    print("="*50)
