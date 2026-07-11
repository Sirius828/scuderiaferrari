#ifndef DIRECTION_RULE_H
#define DIRECTION_RULE_H

#include <string>
#include <vector>

/**
 * @brief 纯规则方向分类，返回 0=直行，1=右转
 * 
 * @param text  OCR 识别出的文本（UTF-8 编码）
 * @return int 0 或 1
 */
int classify_direction(const std::string &text);

#endif // DIRECTION_RULE_H