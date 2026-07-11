#include "direction_rule.h"

#include <algorithm>

// 只去掉 ASCII 空白，避免 std::isspace 误处理 UTF-8 中文字节。
static std::string remove_spaces(const std::string &str) {
    std::string result = str;
    result.erase(std::remove_if(result.begin(), result.end(), [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }), result.end());
    return result;
}

static bool contains_any(const std::string &text, const std::vector<std::string> &keywords) {
    for (const auto &kw : keywords) {
        if (text.find(kw) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static int add_if_contains(const std::string &text,
                           const std::vector<std::string> &keywords,
                           int score) {
    return contains_any(text, keywords) ? score : 0;
}

DirectionRuleResult classify_direction_detail(const std::string &ocr_text) {
    DirectionRuleResult result;
    result.raw_direction = -1;
    result.direction = 0;
    result.rule_score = 0.0f;
    result.right_score = 0;
    result.straight_score = 0;

    std::string t = remove_spaces(ocr_text);
    if (t.empty()) {
        return result;
    }

    std::vector<std::string> neg_words = {
        "禁止", "不能", "不准", "严禁", "别", "不要", "勿", "不许", "不让",
        "杜绝", "不允许", "限制", "切莫", "切勿", "不可"
    };

    std::vector<std::string> right_patterns = {
        "右转", "右拐", "向右转", "向右行驶", "走右边", "走右道",
        "右转弯", "靠右行", "往右转", "往右拐", "抄近道向右", "右侧通行",
        "向右", "往右", "拐右", "右侧"
    };

    std::vector<std::string> straight_patterns = {
        "直行", "直走", "向前走", "前行", "走直道", "继续直行", "直行通过", "一直走"
    };

    std::vector<std::string> near_words = {
        "抄近道", "抄近路", "走近道", "走近路", "近道", "近路", "近道走"
    };

    std::vector<std::string> bad_cond = {
        "封路", "堵死", "不通", "在修", "有事故", "难走", "很难走", "走不了",
        "禁止通行", "施工", "塌方", "封", "堵", "修"
    };

    std::vector<std::string> severe_bad_cond = {
        "封路", "堵死", "不通", "走不了", "禁止通行", "施工", "塌方"
    };

    std::vector<std::string> mild_bad_cond = {
        "难走", "很难走", "有事故", "在修", "堵", "修"
    };

    std::vector<std::string> force_right = {
        "必须右", "只能右", "不得不右", "一定要右", "务必右", "非得右", "只好右",
        "除了右转", "只剩右转"
    };

    std::vector<std::string> force_straight = {
        "必须直", "只能直", "一定要直", "务必直", "非得直", "只好直", "保持直",
        "直道才是", "直道才能", "直道更顺", "直道正确"
    };

    std::vector<std::string> neg_straight_patterns = {
        "禁止直", "不能直", "不准直", "严禁直", "别直", "不要直", "勿直",
        "不许直", "不让直", "切莫直", "切勿直", "不可直"
    };

    std::vector<std::string> irony_words = {
        "才怪", "骗谁", "想多了", "笑话", "别傻", "做梦", "等着迟到"
    };

    const bool has_neg = contains_any(t, neg_words);
    const bool has_right = t.find("右") != std::string::npos;
    const bool has_straight = t.find("直") != std::string::npos || t.find("前") != std::string::npos;
    const bool has_bad = contains_any(t, bad_cond);

    result.right_score += add_if_contains(t, right_patterns, 2);
    result.straight_score += add_if_contains(t, straight_patterns, 2);
    result.right_score += add_if_contains(t, near_words, 4);
    result.right_score += add_if_contains(t, force_right, 5);
    result.straight_score += add_if_contains(t, force_straight, 5);

    if (has_neg && has_right) {
        result.straight_score += 5;
    }
    if (contains_any(t, neg_straight_patterns)) {
        result.right_score += 5;
    }
    if ((t.find("并非禁止右") != std::string::npos) ||
        (t.find("未禁止右") != std::string::npos) ||
        (t.find("并非不让你右") != std::string::npos)) {
        result.right_score += 6;
        result.straight_score -= 3;
    }
    if ((t.find("不得不右") != std::string::npos) ||
        (t.find("不能不右") != std::string::npos) ||
        (t.find("不右转不行") != std::string::npos)) {
        result.right_score += 6;
        result.straight_score -= 3;
    }

    if (contains_any(t, severe_bad_cond) && has_straight) {
        result.right_score += 7;
    } else if (has_bad && has_straight) {
        result.right_score += 5;
    }
    if (contains_any(t, mild_bad_cond) && has_right &&
        (t.find("右道") != std::string::npos || t.find("右边") != std::string::npos)) {
        result.straight_score += 4;
    }
    if (contains_any(t, severe_bad_cond) && has_right && !has_straight &&
        (t.find("右道") != std::string::npos || t.find("右边") != std::string::npos)) {
        result.straight_score += 6;
    }

    if (contains_any(t, irony_words)) {
        if (t.find("右道") != std::string::npos || t.find("右转") != std::string::npos ||
            t.find("右边") != std::string::npos) {
            result.straight_score += 5;
        }
        if ((t.find("直行") != std::string::npos || t.find("直道") != std::string::npos ||
             t.find("直走") != std::string::npos) &&
            !(t.find("右道") != std::string::npos || t.find("右转") != std::string::npos ||
              t.find("右边") != std::string::npos)) {
            result.right_score += 5;
        }
    }

    const int diff = result.right_score - result.straight_score;
    result.rule_score = diff >= 0 ? static_cast<float>(diff) : static_cast<float>(-diff);
    if (diff >= 2) {
        result.raw_direction = 1;
        result.direction = 1;
    } else if (diff <= -2) {
        result.raw_direction = 0;
        result.direction = 0;
    }

    return result;
}

int classify_direction(const std::string &text) {
    return classify_direction_detail(text).direction;
}
