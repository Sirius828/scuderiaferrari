#ifndef DIRECTION_RULE_H
#define DIRECTION_RULE_H

#include <string>
#include <vector>

struct DirectionRuleResult {
    int raw_direction;  // -1=不确定, 0=直行, 1=右转
    int direction;      // 最终输出，unknown 时保守映射为 0
    float rule_score;   // 两类分数差，越大越确定
    int right_score;
    int straight_score;
};

DirectionRuleResult classify_direction_detail(const std::string &text);
int classify_direction(const std::string &text);

#endif // DIRECTION_RULE_H
