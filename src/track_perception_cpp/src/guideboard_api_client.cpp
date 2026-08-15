#include "track_perception_cpp/guideboard_api_client.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <sstream>

#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace track_perception_cpp {
namespace {

size_t writeResponse(char* data, size_t size, size_t count, void* user_data) {
  if (user_data == nullptr) {
    return 0;
  }
  auto* response = static_cast<std::string*>(user_data);
  response->append(data, size * count);
  return size * count;
}

std::string serializeRequest(const std::string& model,
                             const std::vector<GuideboardApiSample>& samples) {
  rapidjson::Document document;
  document.SetObject();
  auto& allocator = document.GetAllocator();

  document.AddMember("model", rapidjson::Value(model.c_str(), allocator), allocator);
  document.AddMember("temperature", 0.0, allocator);
  document.AddMember("stream", false, allocator);

  const char* system_prompt =
      "你是车载路牌OCR纠错与岔路判向器。输入为同一路牌的多次OCR，可能有错字、漏字和多字。"
      "先综合样本恢复原意，再按完整句意选择实际可行或被建议的路线，不得按‘右、直’等单个关键词计数。"
      "本赛道左侧、左走或直走都输出straight，右侧或右走输出right。"
      "若路牌给出左右方向的概率，必须比较数值并选择概率更大的一侧；"
      "例如左侧49.9%、右侧50.1%应输出right。"
      "正确处理否定、反问、反讽和不可通行：不让或不能往右表示straight；直道走不了表示right；"
      "‘右道好走？才怪’表示straight；本赛道‘抄近道’表示right。"
      "只能选择straight或right；无法确定时uncertain=true，但maneuver仍必须填写straight或right，"
      "禁止填写unknown或uncertain。只返回JSON对象，不要Markdown或解释："
      "{\"corrected_text\":\"...\",\"maneuver\":\"straight或right\","
      "\"confidence\":0到1,\"uncertain\":true或false}。";

  rapidjson::Value messages(rapidjson::kArrayType);
  rapidjson::Value system_message(rapidjson::kObjectType);
  system_message.AddMember("role", "system", allocator);
  system_message.AddMember("content", rapidjson::Value(system_prompt, allocator), allocator);
  messages.PushBack(system_message, allocator);

  std::ostringstream user_content;
  user_content << "最近OCR样本（按时间从旧到新）：[";
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i > 0) {
      user_content << ",";
    }
    user_content << "{\"text\":\"";
    for (char ch : samples[i].text) {
      if (ch == '\\' || ch == '"') {
        user_content << '\\' << ch;
      } else if (ch == '\n' || ch == '\r') {
        user_content << ' ';
      } else {
        user_content << ch;
      }
    }
    user_content << "\",\"score\":" << samples[i].score << "}";
  }
  user_content << "]";

  rapidjson::Value user_message(rapidjson::kObjectType);
  user_message.AddMember("role", "user", allocator);
  const std::string content = user_content.str();
  user_message.AddMember("content", rapidjson::Value(content.c_str(), allocator), allocator);
  messages.PushBack(user_message, allocator);
  document.AddMember("messages", messages, allocator);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  document.Accept(writer);
  return buffer.GetString();
}

bool parseDecisionObject(const std::string& json, double uncertain_min_confidence,
                         GuideboardApiResult* result) {
  rapidjson::Document decision;
  decision.Parse(json.c_str());
  if (decision.HasParseError() || !decision.IsObject()) {
    return false;
  }

  const auto maneuver_it = decision.FindMember("maneuver");
  if (maneuver_it == decision.MemberEnd() || !maneuver_it->value.IsString()) {
    return false;
  }
  const std::string maneuver = maneuver_it->value.GetString();
  if (maneuver != "straight" && maneuver != "right") {
    return false;
  }
  result->maneuver = maneuver;

  const auto corrected_it = decision.FindMember("corrected_text");
  if (corrected_it == decision.MemberEnd() || !corrected_it->value.IsString()) {
    result->error = "decision JSON has no corrected_text";
    return false;
  }
  result->corrected_text = corrected_it->value.GetString();
  const auto confidence_it = decision.FindMember("confidence");
  if (confidence_it == decision.MemberEnd() || !confidence_it->value.IsNumber()) {
    result->error = "decision JSON has no confidence";
    return false;
  }
  result->confidence = static_cast<float>(confidence_it->value.GetDouble());
  const auto uncertain_it = decision.FindMember("uncertain");
  if (uncertain_it == decision.MemberEnd() || !uncertain_it->value.IsBool()) {
    result->error = "decision JSON has no uncertain";
    return false;
  }
  result->uncertain = uncertain_it->value.GetBool();
  result->valid = GuideboardApiClient::shouldAcceptDecision(
      result->uncertain, result->confidence, uncertain_min_confidence);
  result->accepted_uncertain = result->valid && result->uncertain;
  if (!result->valid) {
    result->error = "api semantic result is uncertain below confidence threshold";
  }
  return true;
}

bool parseResponse(const std::string& body, double uncertain_min_confidence,
                   GuideboardApiResult* result) {
  rapidjson::Document response;
  response.Parse(body.c_str());
  if (response.HasParseError() || !response.IsObject()) {
    result->error = "response JSON parse failed";
    return false;
  }
  const auto choices_it = response.FindMember("choices");
  if (choices_it == response.MemberEnd() || !choices_it->value.IsArray() ||
      choices_it->value.Empty()) {
    result->error = "response has no choices";
    return false;
  }
  const auto& choice = choices_it->value[0];
  if (!choice.IsObject()) {
    result->error = "response choice is not an object";
    return false;
  }
  const auto message_it = choice.FindMember("message");
  if (message_it == choice.MemberEnd() || !message_it->value.IsObject()) {
    result->error = "response has no message";
    return false;
  }
  const auto content_it = message_it->value.FindMember("content");
  if (content_it == message_it->value.MemberEnd() || !content_it->value.IsString()) {
    result->error = "response message has no text content";
    return false;
  }

  const std::string content = content_it->value.GetString();
  if (parseDecisionObject(content, uncertain_min_confidence, result)) {
    return true;
  }
  const size_t begin = content.find('{');
  const size_t end = content.rfind('}');
  if (begin != std::string::npos && end > begin &&
      parseDecisionObject(content.substr(begin, end - begin + 1),
                          uncertain_min_confidence, result)) {
    return true;
  }
  if (result->error.empty()) {
    result->error = "response content is not the required decision JSON";
  }
  return false;
}

}  // namespace

bool GuideboardApiClient::shouldAcceptDecision(
    bool uncertain, float confidence, double uncertain_min_confidence) {
  if (!uncertain) {
    return true;
  }
  const double threshold = std::clamp(uncertain_min_confidence, 0.0, 1.0);
  return std::isfinite(confidence) && confidence >= threshold && confidence <= 1.0f;
}

GuideboardApiResult GuideboardApiClient::request(
    const std::string& url, const std::string& api_key, const std::string& model,
    const std::vector<GuideboardApiSample>& samples, double timeout_sec,
    double uncertain_min_confidence) {
  GuideboardApiResult result;
  const auto started = std::chrono::steady_clock::now();
  if (url.empty() || api_key.empty()) {
    result.error = api_key.empty() ? "QIANFAN_API_KEY is not set" : "API URL is empty";
    return result;
  }

  static std::once_flag curl_init_once;
  static CURLcode curl_init_result = CURLE_FAILED_INIT;
  std::call_once(curl_init_once, []() {
    curl_init_result = curl_global_init(CURL_GLOBAL_DEFAULT);
  });
  if (curl_init_result != CURLE_OK) {
    result.error = "curl_global_init failed";
    return result;
  }

  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    result.error = "curl_easy_init failed";
    return result;
  }

  const std::string payload = serializeRequest(model, samples);
  std::string response_body;
  const std::string authorization = "Authorization: Bearer " + api_key;
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, authorization.c_str());

  const long timeout_ms = static_cast<long>(std::max(1.0, timeout_sec * 1000.0));
  // TLS through the local proxy occasionally needs more than 500 ms.  The
  // overall request timeout still caps connection plus model inference, so a
  // shorter independent connection deadline only creates premature failures.
  const long connect_timeout_ms = timeout_ms;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeResponse);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, connect_timeout_ms);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

  const CURLcode code = curl_easy_perform(curl);
  result.curl_code = static_cast<int>(code);
  long http_status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);
  result.http_status = static_cast<int>(http_status);
  result.latency_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();

  if (code != CURLE_OK) {
    result.error = curl_easy_strerror(code);
  } else if (http_status < 200 || http_status >= 300) {
    result.error = "HTTP status " + std::to_string(http_status);
  } else {
    result.transport_ok = true;
    parseResponse(response_body, uncertain_min_confidence, &result);
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  return result;
}

}  // namespace track_perception_cpp
