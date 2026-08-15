#pragma once

#include <string>
#include <vector>

namespace track_perception_cpp {

struct GuideboardApiSample {
  std::string text;
  float score{0.0f};
};

struct GuideboardApiResult {
  bool transport_ok{false};
  bool valid{false};
  bool uncertain{false};
  bool accepted_uncertain{false};
  int http_status{0};
  int curl_code{0};
  double latency_ms{-1.0};
  float confidence{0.0f};
  std::string corrected_text;
  std::string maneuver;
  std::string error;
};

class GuideboardApiClient {
 public:
  static bool shouldAcceptDecision(bool uncertain, float confidence,
                                   double uncertain_min_confidence);

  static GuideboardApiResult request(const std::string& url,
                                     const std::string& api_key,
                                     const std::string& model,
                                     const std::vector<GuideboardApiSample>& samples,
                                     double timeout_sec,
                                     double uncertain_min_confidence);
};

}  // namespace track_perception_cpp
