import cv2
import numpy as np
import onnxruntime as ort
import os, sys, time

print("系统启动中（纯规则模式）...", flush=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from utils.db_postprocess import DBPostProcess

# -------------------- 路径配置（仅 OCR） --------------------
DET_MODEL_PATH  = os.path.join(BASE_DIR, "../model/ppocrv4_det.onnx")
REC_MODEL_PATH  = os.path.join(BASE_DIR, "../model/ppocrv4_rec.onnx")
DICT_PATH       = os.path.join(BASE_DIR, "../model/ppocr_keys_v1.txt")
TEST_IMG_PATH   = os.path.join(BASE_DIR, "../model/test27.jpg")
RESULT_IMG_PATH = os.path.join(BASE_DIR, "../result.jpg")

# -------------------- OCR 函数（保持不变） --------------------
def load_dict(dict_path):
    chars = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if line:
                chars.append(line)
    chars.insert(0, '')
    return chars

char_list = load_dict(DICT_PATH)

def det_preprocess(img):
    target_size = 480
    src_h, src_w = img.shape[:2]
    img = cv2.resize(img, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, src_h, src_w, target_size

def rec_preprocess(img):
    h, w = 48, 320
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def rec_decode(preds, chars):
    pred = preds[0]
    argmax_idx = np.argmax(pred, axis=2)[0]
    result = []
    prev, blank = -1, 0
    for idx in argmax_idx:
        idx = int(idx)
        if idx != prev and idx != blank and idx < len(chars):
            result.append(chars[idx])
        prev = idx
    return ''.join(result)

def crop_text_box(img, box):
    x_min = int(np.min(box[:, 0]))
    x_max = int(np.max(box[:, 0]))
    y_min = int(np.min(box[:, 1]))
    y_max = int(np.max(box[:, 1]))
    return img[y_min:y_max, x_min:x_max]

def sort_box_by_y(box):
    return np.mean(box[:, 1])

# -------------------- 加载 OCR 模型 --------------------
ort_option = ort.SessionOptions()
ort_option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_option.intra_op_num_threads = 4
det_session = ort.InferenceSession(DET_MODEL_PATH, sess_options=ort_option, providers=["CPUExecutionProvider"])
rec_session = ort.InferenceSession(REC_MODEL_PATH, sess_options=ort_option, providers=["CPUExecutionProvider"])
det_in_name = det_session.get_inputs()[0].name
rec_in_name = rec_session.get_inputs()[0].name
db_process = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5)
print("OCR 模型加载完成！", flush=True)

# ==================== 纯规则方向分类器 ====================
def classify_direction(text):
    """
    返回 (0 或 1, 置信度)
    0 = 直行， 1 = 右转
    """
    t = text.strip().replace(" ", "")
    if not t:
        return 0, 0.0

    # ---------- 否定词集合 ----------
    neg_words = ["禁止", "不能", "不准", "严禁", "别", "不要", "勿", "不许",
                 "杜绝", "不允许", "限制", "切莫", "切勿", "不可", "并非", "不是"]

    # ---------- 1. 明确的右转指令（无否定） ----------
    right_patterns = ["右转", "右拐", "向右转", "向右行驶", "走右边", "走右道",
                      "右转弯", "靠右行", "往右转", "往右拐", "抄近道向右", "右侧通行"]
    if any(p in t for p in right_patterns) and not any(n in t for n in neg_words):
        return 1, 1.0

    # ---------- 2. 明确的直行指令（无否定） ----------
    straight_patterns = ["直行", "直走", "向前走", "走直道", "继续直行", "直行通过", "一直走"]
    if any(p in t for p in straight_patterns) and not any(n in t for n in neg_words):
        return 0, 1.0

    # ---------- 3. 抄近道/近路系列 → 右转 ----------
    near_words = ["抄近道", "抄近路", "走近道", "走近路", "近道", "近路", "近道走"]
    if any(nw in t for nw in near_words) and not any(n in t for n in neg_words):
        return 1, 1.0

    # ---------- 4. 否定 + 右 → 直行 ----------
    if any(n in t for n in neg_words) and "右" in t:
        return 0, 1.0

    # ---------- 5. 否定 + 直 → 右转 ----------
    if any(n in t for n in neg_words) and "直" in t:
        return 1, 1.0

    # ---------- 6. 转折句：直道负面 → 右转 ----------
    bad_cond = ["封路", "堵死", "不通", "在修", "有事故", "很难走", "走不了", "禁止通行", "施工", "塌方"]
    if any(bc in t for bc in bad_cond) and "直" in t and "右" not in neg_words:
        # 如果直道有负面，且句子没有“禁止右转”等否定，则右转
        if not any(n in t for n in neg_words):
            return 1, 1.0

    # ---------- 7. 右道负面 → 直行 ----------
    if any(bc in t for bc in bad_cond) and "右" in t and "直" in t:
        if "右道" in t or "右边" in t or "右转" not in t:
            return 0, 1.0

    # ---------- 8. 强制/必须右转 ----------
    force_right = ["必须右", "只能右", "不得不右", "一定要右", "务必右", "非得右", "只好右"]
    if any(f in t for f in force_right):
        return 1, 1.0

    # ---------- 9. 默认直行 ----------
    return 0, 0.5   # 低置信度，但安全优先直行

# ==================== 主程序 ====================
if __name__ == "__main__":
    total_start = time.time()

    origin_img = cv2.imread(TEST_IMG_PATH)
    draw_img = origin_img.copy()

    det_input, ori_h, ori_w, resize_s = det_preprocess(origin_img)
    det_out = det_session.run(None, {det_in_name: det_input})
    box_info = db_process({"maps": det_out[0]}, [[ori_h, ori_w, resize_s, resize_s]])
    boxes = box_info[0]["points"]
    boxes = sorted(boxes, key=sort_box_by_y)

    texts = []
    for i, box in enumerate(boxes):
        box_np = np.array(box, np.int32)
        crop = crop_text_box(origin_img, box_np)
        rec_in = rec_preprocess(crop)
        rec_out = rec_session.run(None, {rec_in_name: rec_in})
        text = rec_decode(rec_out, char_list)
        texts.append(text)

        cv2.polylines(draw_img, [box_np], True, (0, 0, 255), 2)
        cv2.putText(draw_img, text, (box_np[0][0], max(12, box_np[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(f"第{i+1}行: {text}")

    cv2.imwrite(RESULT_IMG_PATH, draw_img)
    full_text = "".join(texts)
    print(f"完整文本: {full_text}")

    direction, conf = classify_direction(full_text)
    direction_str = "右转" if direction == 1 else "直行"
    print(f"分类结果: {direction_str} (置信度: {conf:.4f})")
    print(f"总耗时: {time.time() - total_start:.3f} 秒")