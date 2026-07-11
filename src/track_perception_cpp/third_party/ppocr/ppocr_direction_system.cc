#include "ppocr_direction_system.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <exception>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

#include "direction_rule.h"
#include "file_utils.h"
#include "image_utils.h"

#define THRESHOLD 0.3
#define BOX_THRESHOLD 0.6
#define USE_DILATION false
#define DB_SCORE_MODE "slow"
#define DB_BOX_TYPE "poly"
#define DB_UNCLIP_RATIO 1.5

namespace {

const float kLowOcrScore = 0.55f;
const float kEdgeRatio = 0.02f;
const int kEdgePixels = 10;
const int kShortTextChars = 4;

class StdoutSilencer {
public:
    StdoutSilencer() : saved_fd_(-1) {
        std::fflush(stdout);
        saved_fd_ = dup(STDOUT_FILENO);
        int null_fd = open("/dev/null", O_WRONLY);
        if (saved_fd_ >= 0 && null_fd >= 0) {
            dup2(null_fd, STDOUT_FILENO);
        }
        if (null_fd >= 0) {
            close(null_fd);
        }
    }

    ~StdoutSilencer() {
        if (saved_fd_ >= 0) {
            std::fflush(stdout);
            dup2(saved_fd_, STDOUT_FILENO);
            close(saved_fd_);
        }
    }

private:
    int saved_fd_;
};

void reset_result(PPOCRDirectionResult* result) {
    result->text.clear();
    result->direction = 0;
    result->raw_direction = -1;
    result->box_count = 0;
    result->ocr_score = 0.0f;
    result->rule_score = 0.0f;
    result->status = PPOCR_STATUS_OK;
    result->text_touch_edge = false;
    result->time_ms = 0.0;
    result->skipped_box_count = 0;
    result->skipped_box_summary.clear();
    result->error.clear();
}

int utf8_visible_length(const std::string& text) {
    int count = 0;
    for (unsigned char c : text) {
        if (c == '|' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            continue;
        }
        if ((c & 0xC0) != 0x80) {
            ++count;
        }
    }
    return count;
}

bool quad_touches_edge(const rknn_quad_t& box, int width, int height) {
    int min_x = box.left_top.x;
    int max_x = box.left_top.x;
    int min_y = box.left_top.y;
    int max_y = box.left_top.y;

    const rknn_point_t pts[4] = {
        box.left_top, box.right_top, box.right_bottom, box.left_bottom
    };
    for (int i = 0; i < 4; ++i) {
        min_x = std::min(min_x, pts[i].x);
        max_x = std::max(max_x, pts[i].x);
        min_y = std::min(min_y, pts[i].y);
        max_y = std::max(max_y, pts[i].y);
    }

    const int margin_x = std::max(kEdgePixels, static_cast<int>(width * kEdgeRatio));
    const int margin_y = std::max(kEdgePixels, static_cast<int>(height * kEdgeRatio));
    return min_x <= margin_x || min_y <= margin_y ||
           max_x >= width - margin_x || max_y >= height - margin_y;
}

}  // namespace

PPOCRDirectionSystem::PPOCRDirectionSystem() : initialized_(false) {
    std::memset(&ctx_, 0, sizeof(ctx_));
}

PPOCRDirectionSystem::~PPOCRDirectionSystem() {
    release();
}

int PPOCRDirectionSystem::init(const char* det_model_path, const char* rec_model_path) {
    release();

    int ret = 0;
    {
        StdoutSilencer silence;
        ret = init_ppocr_model(det_model_path, &ctx_.det_context);
    }
    if (ret != 0) {
        return ret;
    }

    {
        StdoutSilencer silence;
        ret = init_ppocr_model(rec_model_path, &ctx_.rec_context);
    }
    if (ret != 0) {
        release_ppocr_model(&ctx_.det_context);
        std::memset(&ctx_, 0, sizeof(ctx_));
        return ret;
    }

    initialized_ = true;
    return 0;
}

int PPOCRDirectionSystem::run_image(const char* image_path, PPOCRDirectionResult* result) {
    if (!initialized_ || result == nullptr) {
        return -1;
    }

    reset_result(result);

    image_buffer_t src_image;
    std::memset(&src_image, 0, sizeof(src_image));

    int ret = 0;
    {
        StdoutSilencer silence;
        ret = read_image(image_path, &src_image);
    }
    if (ret != 0) {
        result->status = PPOCR_STATUS_READ_IMAGE_FAILED;
        return ret;
    }

    ret = run_buffer(&src_image, result);

    if (src_image.virt_addr != nullptr) {
        free(src_image.virt_addr);
    }

    return ret;
}

int PPOCRDirectionSystem::run_mat(const cv::Mat& image_rgb, PPOCRDirectionResult* result) {
    if (!initialized_ || result == nullptr) {
        return -1;
    }

    reset_result(result);

    if (image_rgb.empty()) {
        result->status = PPOCR_STATUS_BAD_ARGUMENT;
        return -1;
    }

    cv::Mat rgb;
    if (image_rgb.channels() == 3) {
        rgb = image_rgb.isContinuous() ? image_rgb : image_rgb.clone();
    } else if (image_rgb.channels() == 4) {
        cv::cvtColor(image_rgb, rgb, cv::COLOR_RGBA2RGB);
    } else if (image_rgb.channels() == 1) {
        cv::cvtColor(image_rgb, rgb, cv::COLOR_GRAY2RGB);
    } else {
        result->status = PPOCR_STATUS_BAD_ARGUMENT;
        return -1;
    }

    image_buffer_t src_image;
    std::memset(&src_image, 0, sizeof(src_image));
    src_image.width = rgb.cols;
    src_image.height = rgb.rows;
    src_image.width_stride = rgb.cols;
    src_image.height_stride = rgb.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.virt_addr = rgb.data;
    src_image.size = static_cast<int>(rgb.total() * rgb.elemSize());

    return run_buffer(&src_image, result);
}

int PPOCRDirectionSystem::run_buffer(image_buffer_t* src_image, PPOCRDirectionResult* result) {
    if (!initialized_ || result == nullptr || src_image == nullptr || src_image->virt_addr == nullptr ||
        src_image->width <= 0 || src_image->height <= 0) {
        if (result != nullptr) {
            reset_result(result);
            result->status = PPOCR_STATUS_BAD_ARGUMENT;
        }
        return -1;
    }

    reset_result(result);
    const auto start = std::chrono::steady_clock::now();

    ppocr_det_postprocess_params params;
    params.threshold = THRESHOLD;
    params.box_threshold = BOX_THRESHOLD;
    params.use_dilate = USE_DILATION;
    params.db_score_mode = const_cast<char*>(DB_SCORE_MODE);
    params.db_box_type = const_cast<char*>(DB_BOX_TYPE);
    params.db_unclip_ratio = DB_UNCLIP_RATIO;

    ppocr_text_recog_array_result_t ocr_result;
    std::memset(&ocr_result, 0, sizeof(ocr_result));

    int ret = 0;
    try {
        StdoutSilencer silence;
        ret = inference_ppocr_system_model(&ctx_, src_image, &params, &ocr_result);
    } catch (const cv::Exception& e) {
        ret = -1;
        result->error = e.what();
    } catch (const std::exception& e) {
        ret = -1;
        result->error = e.what();
    } catch (...) {
        ret = -1;
        result->error = "unknown OCR exception";
    }
    result->skipped_box_count = ocr_result.skipped_box_count;
    result->skipped_box_summary = ocr_result.skipped_box_summary;
    if (result->error.empty()) {
        result->error = ocr_result.error;
    }
    if (ret == 0) {
        float score_sum = 0.0f;
        for (int i = 0; i < ocr_result.count; i++) {
            result->text += ocr_result.text_result[i].text.str;
            result->text += "|";
            score_sum += ocr_result.text_result[i].text.score;
            if (quad_touches_edge(ocr_result.text_result[i].box, src_image->width, src_image->height)) {
                result->text_touch_edge = true;
            }
        }
        result->box_count = ocr_result.count;
        result->ocr_score = ocr_result.count > 0 ? score_sum / ocr_result.count : 0.0f;

        DirectionRuleResult rule = classify_direction_detail(result->text);
        result->raw_direction = rule.raw_direction;
        result->direction = rule.direction;
        result->rule_score = rule.rule_score;

        if (result->box_count == 0) {
            result->status = PPOCR_STATUS_NO_TEXT;
            result->raw_direction = -1;
            result->direction = 0;
        } else if (result->text_touch_edge && utf8_visible_length(result->text) <= kShortTextChars) {
            result->status = PPOCR_STATUS_SHORT_TEXT_WITH_EDGE_RISK;
            result->raw_direction = -1;
            result->direction = 0;
        } else if (result->ocr_score > 0.0f && result->ocr_score < kLowOcrScore) {
            result->status = PPOCR_STATUS_LOW_OCR_SCORE;
        } else if (result->raw_direction == -1) {
            result->status = PPOCR_STATUS_UNCERTAIN_RULE;
            result->direction = 0;
        } else if (result->text_touch_edge) {
            result->status = PPOCR_STATUS_TEXT_TOUCH_EDGE;
        } else {
            result->status = PPOCR_STATUS_OK;
        }
    } else {
        result->status = PPOCR_STATUS_INFERENCE_FAILED;
    }

    const auto end = std::chrono::steady_clock::now();
    result->time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ret;
}

void PPOCRDirectionSystem::release() {
    if (!initialized_) {
        return;
    }

    release_ppocr_model(&ctx_.det_context);
    release_ppocr_model(&ctx_.rec_context);
    std::memset(&ctx_, 0, sizeof(ctx_));
    initialized_ = false;
}
