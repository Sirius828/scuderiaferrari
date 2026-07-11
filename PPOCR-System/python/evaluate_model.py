import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, "../model/ernie-3.0-nano-zh"))
sess = ort.InferenceSession("ernie_direction.onnx")

def predict(text):
    enc = tokenizer(text, truncation=True, max_length=128, return_tensors="np", padding=True)
    logits = sess.run(None, {"input_ids": enc['input_ids'].astype(np.int64),
                             "token_type_ids": enc['token_type_ids'].astype(np.int64)})[0]
    return "右转" if np.argmax(logits[0]) == 1 else "直行"

# 读取测试用例文件
test_file = "test_cases.txt"
with open(test_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

errors = []
correct = 0
total = 0

for line in lines:
    parts = line.strip().split("\t")
    if len(parts) != 2:
        continue
    text, expected = parts
    predicted = predict(text)
    total += 1
    if predicted == expected:
        correct += 1
    else:
        errors.append((text, expected, predicted))

print(f"总测试: {total}, 正确: {correct}, 错误: {len(errors)}")
print(f"当前准确率: {correct/total*100:.2f}%")
if errors:
    print("\n========== 错误样例 ==========")
    for text, exp, pred in errors[:20]:   # 最多显示20条
        print(f"  预期:{exp} 预测:{pred} | {text}")
    # 将错误句子保存，方便后续加入训练
    with open("error_samples.txt", "w", encoding="utf-8") as ef:
        for text, exp, pred in errors:
            ef.write(f"{text}\t{exp}\n")
    print(f"错误句子已保存至 error_samples.txt")
    