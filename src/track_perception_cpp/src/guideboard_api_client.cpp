#include "track_perception_cpp/guideboard_api_client.hpp"

#include <algorithm>
#include <chrono>
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
      "你是车载路牌文字纠错和岔路方向判定器。输入是同一个路牌最近几次本地OCR结果，OCR可能有错别字、漏字或多余字。"
      "请结合所有样本纠正文字并判断这张路牌对应的岔路决策。只能返回 straight 或 right："
      "straight 表示直走，right 表示右转。不要根据不存在的信息猜测左转；如果无法确定，设置 uncertain=true。"
      "必须严格返回一个JSON对象，不要Markdown，不要解释，字段必须是 corrected_text(string)、"
      "maneuver(straight/right)、confidence(0到1的小数)、uncertain(boolean)。";

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

bool parseDecisionObject(const std::string& json, GuideboardApiResult* result) {
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
  result->valid = !result->uncertain;
  if (!result->valid) {
    result->error = "api semantic result is uncertain";
  }
  return true;
}

bool parseResponse(const std::string& body, GuideboardApiResult* result) {
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
  if (parseDecisionObject(content, result)) {
    return true;
  }
  const size_t begin = content.find('{');
  const size_t end = content.rfind('}');
  if (begin != std::string::npos && end > begin &&
      parseDecisionObject(content.substr(begin, end - begin + 1), result)) {
    return true;
  }
  if (result->error.empty()) {
    result->error = "response content is not the required decision JSON";
  }
  return false;
}

}  // namespace

GuideboardApiResult GuideboardApiClient::request(
    const std::string& url, const std::string& api_key, const std::string& model,
    const std::vector<GuideboardApiSample>& samples, double timeout_sec) {
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
  const long connect_timeout_ms = std::min(500L, timeout_ms);
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
    parseResponse(response_body, &result);
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  return result;
}

}  // namespace track_perception_cpp
