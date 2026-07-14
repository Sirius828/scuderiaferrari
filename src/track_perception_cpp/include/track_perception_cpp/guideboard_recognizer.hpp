#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace track_perception_cpp {

struct GuideboardTemplateSpec {
  std::string id;
  std::string text;
  std::string maneuver;
};

struct GuideboardRecognizerConfig {
  float min_text_score{0.35f};
  float evidence_decay{0.7f};
  float stable_min_evidence{4.0f};
  float stable_min_margin{2.0f};
  int stable_frames{2};
  int min_unique_bigram_hits{2};
};

struct GuideboardMatch {
  bool eligible{false};
  bool stable{false};
  int best_index{-1};
  std::string normalized_text;
  std::string best_id;
  std::string maneuver{"unknown"};
  std::string reason;
  float model_score{0.0f};
  float best_score{0.0f};
  float margin{0.0f};
  int consecutive_wins{0};
  std::vector<int> unique_bigram_hits;
  std::vector<float> frame_scores;
  std::vector<float> evidence;
};

class GuideboardRecognizer {
 public:
  bool configure(const std::vector<GuideboardTemplateSpec>& templates,
                 const GuideboardRecognizerConfig& config,
                 std::string* error = nullptr);
  void reset();
  GuideboardMatch update(const std::string& ocr_text, float model_score);

  const std::vector<GuideboardTemplateSpec>& templates() const { return templates_; }
  const GuideboardMatch& lastMatch() const { return last_match_; }

  static std::string normalizeUtf8(const std::string& text);
  static float normalizedLcsRatio(const std::string& a, const std::string& b);

 private:
  using Codepoints = std::u32string;
  using Bigram = uint64_t;

  static Codepoints decodeUtf8(const std::string& text);
  static std::string encodeUtf8(const Codepoints& text);
  static std::unordered_set<Bigram> makeBigrams(const Codepoints& text);
  static size_t lcsLength(const Codepoints& a, const Codepoints& b);
  GuideboardMatch makeBaseMatch(const std::string& normalized_text, float model_score) const;

  GuideboardRecognizerConfig config_;
  std::vector<GuideboardTemplateSpec> templates_;
  std::vector<Codepoints> normalized_templates_;
  std::vector<std::unordered_set<Bigram>> unique_bigrams_;
  std::vector<float> evidence_;
  int consecutive_winner_{-1};
  int consecutive_wins_{0};
  int stable_index_{-1};
  GuideboardMatch last_match_;
};

}  // namespace track_perception_cpp
