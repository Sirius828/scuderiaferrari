#include "direction_rule.h"
#include <algorithm>  // for std::find_if

// 只去掉 ASCII 空白，避免 std::isspace 误处理 UTF-8 中文字节。
static std::string remove_spaces(const std::string &str) {
    std::string result = str;
    result.erase(std::remove_if(result.begin(), result.end(), [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }), result.end());
    return result;
}

// 检查字符串中是否包含任意关键词
static bool contains_any(const std::string &text, const std::vector<std::string> &keywords) {
    for (const auto &kw : keywords) {
        if (text.find(kw) != std::string::npos) {
            return true;
        }
    }
    return false;
}

int classify_direction(const std::string &ocr_text) {
    // 去掉空格
    std::string t = remove_spaces(ocr_text);
    if (t.empty()) return 0;  // 默认直行

    // 否定词集合（UTF-8 编码，直接写中文字符串）
    std::vector<std::string> neg_words = {
        "禁止", "不能", "不准", "严禁", "别", "不要", "勿", "不许",
        "杜绝", "不允许", "限制", "切莫", "切勿", "不可", "并非", "不是"
    };

    // 1. 明确的右转指令（无否定）
    std::vector<std::string> right_patterns = {
        "右转", "右拐", "向右转", "向右行驶", "走右边", "走右道",
        "右转弯", "靠右行", "往右转", "往右拐", "抄近道向右", "右侧通行"
    };
    if (contains_any(t, right_patterns) && !contains_any(t, neg_words)) {
        return 1;
    }

    // 2. 明确的直行指令（无否定）
    std::vector<std::string> straight_patterns = {
        "直行", "直走", "向前走", "走直道", "继续直行", "直行通过", "一直走"
    };
    if (contains_any(t, straight_patterns) && !contains_any(t, neg_words)) {
        return 0;
    }

    // 3. 抄近道/近路 → 右转
    std::vector<std::string> near_words = {
        "抄近道", "抄近路", "走近道", "走近路", "近道", "近路", "近道走"
    };
    if (contains_any(t, near_words) && !contains_any(t, neg_words)) {
        return 1;
    }

    // 4. 否定 + 右 → 直行
    if (contains_any(t, neg_words) && t.find("右") != std::string::npos) {
        return 0;
    }

    // 5. 否定 + 直 → 右转
    if (contains_any(t, neg_words) && t.find("直") != std::string::npos) {
        return 1;
    }

    // 6. 转折句：直道负面 → 右转
    std::vector<std::string> bad_cond = {
        "封路", "堵死", "不通", "在修", "有事故", "难走", "很难走", "走不了",
        "禁止通行", "施工", "塌方"
    };
    if (contains_any(t, bad_cond) && t.find("直") != std::string::npos &&
        !contains_any(t, neg_words)) {
        return 1;
    }

    // 7. 右道负面 → 直行
    if (contains_any(t, bad_cond) && t.find("右") != std::string::npos &&
        t.find("直") != std::string::npos) {
        if (t.find("右道") != std::string::npos || t.find("右边") != std::string::npos ||
            t.find("右转") == std::string::npos) {
            return 0;
        }
    }

    // 8. 强制/必须右转
    std::vector<std::string> force_right = {
        "必须右", "只能右", "不得不右", "一定要右", "务必右", "非得右", "只好右"
    };
    if (contains_any(t, force_right)) {
        return 1;
    }

    // 9. 默认直行
    return 0;
}
