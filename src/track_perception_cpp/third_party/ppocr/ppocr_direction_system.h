#ifndef PPOCR_DIRECTION_SYSTEM_H
#define PPOCR_DIRECTION_SYSTEM_H

#include <string>

#include <opencv2/core.hpp>

#include "ppocr_system.h"

enum PPOCRDirectionStatus {
    PPOCR_STATUS_OK = 0,
    PPOCR_STATUS_NO_TEXT = 1,
    PPOCR_STATUS_LOW_OCR_SCORE = 2,
    PPOCR_STATUS_UNCERTAIN_RULE = 3,
    PPOCR_STATUS_TEXT_TOUCH_EDGE = 4,
    PPOCR_STATUS_SHORT_TEXT_WITH_EDGE_RISK = 5,
    PPOCR_STATUS_BAD_ARGUMENT = 6,
    PPOCR_STATUS_READ_IMAGE_FAILED = 7,
    PPOCR_STATUS_INFERENCE_FAILED = 8
};

struct PPOCRDirectionResult {
    std::string text;
    int direction;       // 最终输出：0=直行，1=右转
    int raw_direction;   // 内部判断：-1=不确定，0=直行，1=右转
    int box_count;
    float ocr_score;
    float rule_score;
    int status;
    bool text_touch_edge;
    double time_ms;
    int skipped_box_count;
    std::string skipped_box_summary;
    std::string error;
};

class PPOCRDirectionSystem {
public:
    PPOCRDirectionSystem();
    ~PPOCRDirectionSystem();

    int init(const char* det_model_path, const char* rec_model_path);
    int init_rec(const char* rec_model_path);
    int run_image(const char* image_path, PPOCRDirectionResult* result);
    int run_mat(const cv::Mat& image_rgb, PPOCRDirectionResult* result);
    int run_det_rec_mat(const cv::Mat& image_rgb, PPOCRDirectionResult* result);
    int run_rec_mat(const cv::Mat& image_rgb, PPOCRDirectionResult* result);
    void release();

private:
    int run_buffer(image_buffer_t* src_image, PPOCRDirectionResult* result);

    ppocr_system_app_context ctx_;
    bool det_initialized_;
    bool rec_initialized_;
};

#endif
