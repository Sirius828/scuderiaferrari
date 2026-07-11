# rule_based.py — 测试规则引擎

def predict_direction(text):
    text = text.replace(" ", "").replace("\n", "")
    neg_words = ["禁止", "不许", "不准", "严禁", "不能", "不要", "别", "勿", "杜绝", "禁止"]
    force_words = ["必须", "只能", "非得", "不得不", "绝对", "唯一"]
    irony_words = ["才怪", "骗谁", "想多了", "别傻了", "笑话", "骗谁呢", "才不", "才不是"]

    # 1. 否定 + 右 → 直行
    for neg in neg_words:
        if neg in text and "右" in text:
            # 但如果是强制否定？ 比如“不得不右转” — “不得不”已经在 force_words 里
            if "不得不" in text or "不能不" in text:
                return "右转"
            return "直行"

    # 2. 强制 + 右 → 右转
    for force in force_words:
        if force in text and "右" in text:
            return "右转"

    # 3. 直行关键词
    if "直" in text:
        return "直行"

    # 4. 右转关键词
    if "右" in text:
        return "右转"

    # 5. 反讽反转
    for irony in irony_words:
        if irony in text:
            # 如果前面已经有方向预判，但这里直接按反义词组默认处理
            if "直" in text:
                return "右转"
            elif "右" in text:
                return "直行"

    # 默认直行
    return "直行"


# 测试
tests = [
    ("右道真的有点崎岖，但是直道真的走不了", "右转"),
    ("右道比直道好走多了？才怪！", "直行"),
    ("他们来电话了，说不让我方向盘往右打！", "直行"),
    ("我就说三个字：抄近道", "右转"),
    ("请走右侧车道", "右转"),
    ("前方禁止右转", "直行"),
    ("右边堵死了，还不如直走", "直行"),
    ("我偏要右拐", "右转"),
    ("现在必须抄近道，走右边", "右转"),
    ("别右转，前面封路了", "直行"),
    ("右转省时间？骗谁呢", "直行"),
    ("直行快？别傻了", "右转"),
    ("不能右转弯", "直行"),
    ("必须直走", "直行"),
]
for text, exp in tests:
    res = predict_direction(text)
    status = "✓" if res == exp else "✗"
    print(f"{status} 预期:{exp} 预测:{res} | {text}")