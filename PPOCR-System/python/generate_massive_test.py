import random
import itertools

# ======================== 词汇库 ========================
# 右转相关词
right_verbs = ["右转", "右拐", "向右转", "向右行驶", "走右边", "走右道", "右转弯", "靠右行", "抄近道向右"]
# 直行相关词
straight_verbs = ["直行", "直走", "向前走", "走直道", "禁止右转后的替代路径", "继续直行"]
# 否定词
neg_words = ["不", "禁止", "不能", "不准", "不许", "严禁", "别", "不要", "勿", "杜绝"]
# 程度副词
emph_words = ["一定", "必须", "只能", "非得", "绝对", "唯一选择是", "除了...别无他路"]
# 路面情况
road_conditions = ["堵死了", "封路了", "在修路", "有事故", "很崎岖", "不好走", "畅通无阻", "特别好走"]
# 比较级
compare_words = ["比...好走多了", "比...快多了", "不如", "还不如", "省时间"]
# 口语化表达
colloquial = ["我跟你说", "听着", "就三个字", "他们说", "导航说", "前面有人喊"]

# ======================== 模板构造函数 ========================
def generate_sentences():
    sentences = []
    
    # 1. 简单祈使句
    for rv in right_verbs:
        sentences.append((f"{rv}", "右转"))
        sentences.append((f"请{rv}", "右转"))
        sentences.append((f"建议{rv}", "右转"))
    for sv in straight_verbs:
        sentences.append((f"{sv}", "直行"))
        sentences.append((f"请{sv}", "直行"))
    
    # 2. 否定祈使句
    for neg in neg_words:
        for rv in right_verbs:
            sentences.append((f"{neg}{rv}", "直行"))
            sentences.append((f"{neg}要{rv}", "直行"))
            sentences.append((f"{neg}可以{rv}", "直行"))
            sentences.append((f"前方{neg}{rv}", "直行"))
        for sv in straight_verbs:
            # 否定直行？实际可能暗示右转？但这类句子少见，先不加
            pass
    
    # 3. 复合句（转折/条件）
    patterns_right = [
        ("右道{t}, 所以走右边", "右转"),
        ("{neg}右转的话，只能直行了", "直行"),
        ("如果右道{t}, 就右拐", "右转"),
        ("右转{t}吗？那就走吧", "右转"),
        ("虽然右道{t}, 但必须右转", "右转"),
        ("{neg}右转，那边{t}", "直行"),
        ("除非右道{t}, 否则不要右转", "直行"),
        ("要是右边{t}, 就直走算了", "直行"),
    ]
    # t 代表路况，neg代表否定
    for templ, label in patterns_right:
        for cond in road_conditions:
            for neg in neg_words:
                s = templ.format(t=cond, neg=neg)
                sentences.append((s, label))
    
    # 4. 反讽/反问句
    irony_templates = [
        ("右道好走？才怪！", "直行"),
        ("右转省时间？骗谁呢", "直行"),
        ("你以为右边快？想多了", "直行"),
        ("右道比直道好走多了？才怪", "直行"),
        ("直行快？别傻了", "右转"),
        ("走直道？等着迟到吧", "右转"),
        ("右转就能快？笑话", "直行"),
    ]
    for s, label in irony_templates:
        sentences.append((s, label))
    
    # 5. 强调句型
    emph_right = ["必须右转", "只能走右边", "非得右拐", "绝对要右转"]
    for s in emph_right:
        sentences.append((s, "右转"))
    emph_straight = ["必须直行", "只能直走", "非得直行"]
    for s in emph_straight:
        sentences.append((s, "直行"))
    
    # 6. 口语化 + 间接表达
    collo_right = [
        ("我跟你说，走右边准没错", "右转"),
        ("听哥的，右转", "右转"),
        ("导航说了，前面右拐", "右转"),
        ("前面有人喊往右", "右转"),
    ]
    for s, label in collo_right:
        sentences.append((s, label))
    collo_straight = [
        ("他们说不能右转", "直行"),
        ("交警让我直走", "直行"),
        ("别听导航瞎说右转", "直行"),
    ]
    for s, label in collo_straight:
        sentences.append((s, label))
    
    # 7. 双重否定等复杂句式
    double_neg = [
        ("不右转不行了", "右转"),        # 必须右转
        ("没有不右转的道理", "右转"),
        ("不得不右转", "右转"),
        ("不能不走右边", "右转"),
        ("不是不能直行，而是必须右转", "右转"),
        ("并非禁止右转", "右转"),
    ]
    for s, label in double_neg:
        sentences.append((s, label))
    
    # 去重（以防模板重叠产生重复）
    seen = set()
    unique = []
    for s, l in sentences:
        if s not in seen:
            seen.add(s)
            unique.append((s, l))
    return unique

# 生成全部句子
all_cases = generate_sentences()
print(f"生成测试句子总数: {len(all_cases)}")

# 保存到文件 (可选)
with open("test_cases.txt", "w", encoding="utf-8") as f:
    for s, l in all_cases:
        f.write(f"{s}\t{l}\n")
print("已保存到 test_cases.txt")
