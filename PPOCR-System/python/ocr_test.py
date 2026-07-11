import cv2
import numpy as np
import onnxruntime as ort
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from utils.db_postprocess import DBPostProcess

# 路径完全沿用你的项目结构
DET_MODEL_PATH = os.path.join(BASE_DIR, "../model/ppocrv4_det.onnx")
REC_MODEL_PATH = os.path.join(BASE_DIR, "../model/ppocrv4_rec.onnx")
DICT_PATH = os.path.join(BASE_DIR, "../model/ppocr_keys_v1.txt")
TEST_IMG_PATH = os.path.join(BASE_DIR, "../model/test.jpg")
RESULT_IMG_PATH = os.path.join(BASE_DIR, "../result.jpg")

# 原版字典加载，严格过滤空行，保证索引和模型训练完全一致
def load_dict(dict_path):
    """加载字典，并保证索引 0 为 blank（用于跳过），真实字符从索引 1 开始"""
    chars = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if line:
                chars.append(line)
    # 关键：在列表头部插入一个空字符串，位置 0 对应 CTC blank
    # 这样 chars[1] 才对应该字典的第一个真实字符
    chars.insert(0, '')
    return chars

# 【原版PPOCRv4官方检测预处理，无任何自定义修改】
def det_preprocess(img):
    h, w = img.shape[:2]
    target_size = 480
    img = cv2.resize(img, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, h, w, target_size

# 【原版识别预处理，无修改】
def rec_preprocess(img):
    target_h, target_w = 48, 320
    img = cv2.resize(img, (target_w, target_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # 转为 RGB
    img = img.astype(np.float32)
    img /= 255.0                             # 归一化到 [0, 1]
    # 如果你的 ONNX 模型确实需要归一到 [-1,1]，则替换为下面注释的代码
    # img = (img - 0.5) / 0.5
    img = img.transpose((2, 0, 1))           # HWC -> CHW
    img = np.expand_dims(img, axis=0)         # (1,3,48,320)
    return img.astype(np.float32)

# 原版解码函数
def rec_decode(preds, chars):
    """CTC 贪心解码，blank 索引为 0"""
    pred = preds[0]                    # (1, T, num_classes)
    argmax_idx = np.argmax(pred, axis=2)[0]  # (T,)
    result = []
    prev = -1
    blank = 0
    for idx in argmax_idx:
        idx = int(idx)
        if idx != prev and idx != blank and idx < len(chars):
            result.append(chars[idx])   # idx=1 取 chars[1]，对应真实首字符
        prev = idx
    return ''.join(result)

def crop_box(img, box):
    x1 = int(np.min(box[:, 0]))
    x2 = int(np.max(box[:, 0]))
    y1 = int(np.min(box[:, 1]))
    y2 = int(np.max(box[:, 1]))
    return img[y1:y2, x1:x2]

def sort_by_y(box):
    return np.mean(box[:, 1])

if __name__ == "__main__":
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 4
    det_sess = ort.InferenceSession(DET_MODEL_PATH, opt, ["CPUExecutionProvider"])
    rec_sess = ort.InferenceSession(REC_MODEL_PATH, opt, ["CPUExecutionProvider"])
    det_in_name = det_sess.get_inputs()[0].name
    rec_in_name = rec_sess.get_inputs()[0].name

    db = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5)
    chars = load_dict(DICT_PATH)

    img = cv2.imread(TEST_IMG_PATH)
    draw = img.copy()

    det_in, orig_h, orig_w, resize_s = det_preprocess(img)
    det_out = det_sess.run(None, {det_in_name: det_in})
    pred_map = det_out[0]
    boxes_info = db({"maps": pred_map}, [[orig_h, orig_w, resize_s, resize_s]])
    boxes = boxes_info[0]["points"]
    boxes = sorted(boxes, key=sort_by_y)

    text_list = []
    for b in boxes:
        b_np = np.array(b, np.int32)
        crop = crop_box(img, b_np)
        rec_in = rec_preprocess(crop)
        rec_out = rec_sess.run(None, {rec_in_name: rec_in})
        text = rec_decode(rec_out, chars)
        text_list.append(text)
        cv2.polylines(draw, [b_np], True, (0,0,255), 2)
        print("识别文字：", text)

    cv2.imwrite(RESULT_IMG_PATH, draw)
    print("全部识别文本：", text_list)
