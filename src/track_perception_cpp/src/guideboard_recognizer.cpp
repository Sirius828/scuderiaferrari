#include "track_perception_cpp/guideboard_recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace track_perception_cpp {

namespace {

bool isDiscardedCodepoint(char32_t cp) {
  if (cp <= 0x7f) {
    return cp <= 0x20 || cp == 0x7f ||
           cp == '!' || cp == '"' || cp == '#' || cp == '$' || cp == '%' || cp == '&' ||
           cp == '\'' || cp == '(' || cp == ')' || cp == '*' || cp == '+' || cp == ',' ||
           cp == '-' || cp == '.' || cp == '/' || cp == ':' || cp == ';' || cp == '<' ||
           cp == '=' || cp == '>' || cp == '?' || cp == '@' || cp == '[' || cp == '\\' ||
           cp == ']' || cp == '^' || cp == '_' || cp == '`' || cp == '{' || cp == '|' ||
           cp == '}' || cp == '~';
  }
  switch (cp) {
    case U'　': case U'，': case U'。': case U'！': case U'？': case U'：': case U'；':
    case U'、': case U'“': case U'”': case U'‘': case U'’': case U'（': case U'）':
    case U'【': case U'】': case U'《': case U'》': case U'…': case U'—': case U'·':
      return true;
    default:
      return false;
  }
}

}  // namespace

GuideboardRecognizer::Codepoints GuideboardRecognizer::decodeUtf8(const std::string& text) {
  Codepoints out;
  for (size_t i = 0; i < text.size();) {
    const unsigned char lead = static_cast<unsigned char>(text[i]);
    char32_t cp = 0xfffd;
    size_t count = 1;
    if (lead < 0x80) {
      cp = lead;
    } else if ((lead & 0xe0) == 0xc0 && i + 1 < text.size()) {
      cp = lead & 0x1f;
      count = 2;
    } else if ((lead & 0xf0) == 0xe0 && i + 2 < text.size()) {
      cp = lead & 0x0f;
      count = 3;
    } else if ((lead & 0xf8) == 0xf0 && i + 3 < text.size()) {
      cp = lead & 0x07;
      count = 4;
    }
    bool valid = count > 1;
    for (size_t j = 1; j < count; ++j) {
      const unsigned char continuation = static_cast<unsigned char>(text[i + j]);
      if ((continuation & 0xc0) != 0x80) {
        valid = false;
        count = 1;
        cp = 0xfffd;
        break;
      }
      cp = (cp << 6) | (continuation & 0x3f);
    }
    if (lead < 0x80) {
      valid = true;
    }
    out.push_back(valid ? cp : 0xfffd);
    i += count;
  }
  return out;
}

std::string GuideboardRecognizer::encodeUtf8(const Codepoints& text) {
  std::string out;
  for (char32_t cp : text) {
    if (cp <= 0x7f) {
      out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
      out.push_back(static_cast<char>(0xc0 | ((cp >> 6) & 0x1f)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
      out.push_back(static_cast<char>(0xe0 | ((cp >> 12) & 0x0f)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
      out.push_back(static_cast<char>(0xf0 | ((cp >> 18) & 0x07)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
  }
  return out;
}

std::string GuideboardRecognizer::normalizeUtf8(const std::string& text) {
  Codepoints normalized;
  for (char32_t cp : decodeUtf8(text)) {
    if (cp >= 0xff01 && cp <= 0xff5e) {
      cp -= 0xfee0;
    }
    if (!isDiscardedCodepoint(cp) && cp != 0xfffd) {
      normalized.push_back(cp);
    }
  }
  return encodeUtf8(normalized);
}

std::unordered_set<GuideboardRecognizer::Bigram> GuideboardRecognizer::makeBigrams(
    const Codepoints& text) {
  std::unordered_set<Bigram> result;
  for (size_t i = 1; i < text.size(); ++i) {
    result.insert((static_cast<Bigram>(text[i - 1]) << 32) | static_cast<uint32_t>(text[i]));
  }
  return result;
}

size_t GuideboardRecognizer::lcsLength(const Codepoints& a, const Codepoints& b) {
  if (a.empty() || b.empty()) {
    return 0;
  }
  const Codepoints* rows = &a;
  const Codepoints* cols = &b;
  if (cols->size() > rows->size()) {
    std::swap(rows, cols);
  }
  std::vector<size_t> previous(cols->size() + 1, 0);
  std::vector<size_t> current(cols->size() + 1, 0);
  for (char32_t row_cp : *rows) {
    for (size_t j = 1; j <= cols->size(); ++j) {
      current[j] = row_cp == (*cols)[j - 1]
                       ? previous[j - 1] + 1
                       : std::max(previous[j], current[j - 1]);
    }
    std::swap(previous, current);
    std::fill(current.begin(), current.end(), 0);
  }
  return previous.back();
}

float GuideboardRecognizer::normalizedLcsRatio(const std::string& a, const std::string& b) {
  const Codepoints normalized_a = decodeUtf8(normalizeUtf8(a));
  const Codepoints normalized_b = decodeUtf8(normalizeUtf8(b));
  const size_t denominator = std::min(normalized_a.size(), normalized_b.size());
  return denominator > 0
             ? static_cast<float>(lcsLength(normalized_a, normalized_b)) /
                   static_cast<float>(denominator)
             : 0.0f;
}

bool GuideboardRecognizer::configure(const std::vector<GuideboardTemplateSpec>& templates,
                                     const GuideboardRecognizerConfig& config,
                                     std::string* error) {
  if (templates.size() < 2) {
    if (error != nullptr) {
      *error = "at least two guideboard templates are required";
    }
    return false;
  }
  std::unordered_set<std::string> ids;
  std::vector<Codepoints> normalized;
  normalized.reserve(templates.size());
  for (const auto& spec : templates) {
    if (spec.id.empty() || !ids.insert(spec.id).second) {
      if (error != nullptr) {
        *error = "guideboard template ids must be non-empty and unique";
      }
      return false;
    }
    if (spec.maneuver != "straight" && spec.maneuver != "right") {
      if (error != nullptr) {
        *error = "guideboard maneuver must be straight or right";
      }
      return false;
    }
    Codepoints text = decodeUtf8(normalizeUtf8(spec.text));
    if (text.size() < 2) {
      if (error != nullptr) {
        *error = "guideboard template text is too short";
      }
      return false;
    }
    normalized.push_back(std::move(text));
  }

  std::vector<std::unordered_set<Bigram>> all_bigrams;
  all_bigrams.reserve(normalized.size());
  std::unordered_map<Bigram, int> owners;
  for (const auto& text : normalized) {
    all_bigrams.push_back(makeBigrams(text));
    for (Bigram gram : all_bigrams.back()) {
      ++owners[gram];
    }
  }
  std::vector<std::unordered_set<Bigram>> unique_bigrams(all_bigrams.size());
  for (size_t i = 0; i < all_bigrams.size(); ++i) {
    for (Bigram gram : all_bigrams[i]) {
      if (owners[gram] == 1) {
        unique_bigrams[i].insert(gram);
      }
    }
    if (unique_bigrams[i].empty()) {
      if (error != nullptr) {
        *error = "each guideboard template needs at least one unique bigram";
      }
      return false;
    }
  }

  config_ = config;
  config_.min_text_score = std::clamp(config_.min_text_score, 0.0f, 1.0f);
  config_.evidence_decay = std::clamp(config_.evidence_decay, 0.0f, 0.999f);
  config_.stable_min_evidence = std::max(0.0f, config_.stable_min_evidence);
  config_.stable_min_margin = std::max(0.0f, config_.stable_min_margin);
  config_.stable_frames = std::max(1, config_.stable_frames);
  config_.min_unique_bigram_hits = std::max(1, config_.min_unique_bigram_hits);
  templates_ = templates;
  normalized_templates_ = std::move(normalized);
  unique_bigrams_ = std::move(unique_bigrams);
  reset();
  return true;
}

void GuideboardRecognizer::reset() {
  evidence_.assign(templates_.size(), 0.0f);
  consecutive_winner_ = -1;
  consecutive_wins_ = 0;
  stable_index_ = -1;
  last_match_ = makeBaseMatch("", 0.0f);
  last_match_.reason = "reset";
}

GuideboardMatch GuideboardRecognizer::makeBaseMatch(const std::string& normalized_text,
                                                    float model_score) const {
  GuideboardMatch match;
  match.normalized_text = normalized_text;
  match.model_score = model_score;
  match.unique_bigram_hits.assign(templates_.size(), 0);
  match.frame_scores.assign(templates_.size(), 0.0f);
  match.evidence = evidence_;
  return match;
}

GuideboardMatch GuideboardRecognizer::update(const std::string& ocr_text, float model_score) {
  const std::string normalized_text = normalizeUtf8(ocr_text);
  GuideboardMatch match = makeBaseMatch(normalized_text, model_score);
  if (stable_index_ >= 0) {
    match.stable = true;
    match.best_index = stable_index_;
    match.best_id = templates_[stable_index_].id;
    match.maneuver = templates_[stable_index_].maneuver;
    match.evidence = evidence_;
    match.reason = "already_stable";
    last_match_ = match;
    return match;
  }
  if (normalized_text.empty()) {
    match.reason = "empty_text";
    last_match_ = match;
    return match;
  }
  if (model_score < config_.min_text_score) {
    match.reason = "low_model_score";
    last_match_ = match;
    return match;
  }

  const Codepoints observed = decodeUtf8(normalized_text);
  const auto observed_bigrams = makeBigrams(observed);
  int frame_best = -1;
  float frame_best_score = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < templates_.size(); ++i) {
    int hits = 0;
    for (Bigram gram : unique_bigrams_[i]) {
      hits += observed_bigrams.count(gram) != 0 ? 1 : 0;
    }
    const size_t denominator = std::min(observed.size(), normalized_templates_[i].size());
    const float lcs_score = denominator > 0
                                ? static_cast<float>(lcsLength(observed, normalized_templates_[i])) /
                                      static_cast<float>(denominator)
                                : 0.0f;
    match.unique_bigram_hits[i] = hits;
    match.frame_scores[i] = static_cast<float>(hits) + lcs_score;
    if (match.frame_scores[i] > frame_best_score) {
      frame_best_score = match.frame_scores[i];
      frame_best = static_cast<int>(i);
    }
  }

  if (frame_best < 0 ||
      match.unique_bigram_hits[frame_best] < config_.min_unique_bigram_hits) {
    match.best_index = frame_best;
    if (frame_best >= 0) {
      match.best_id = templates_[frame_best].id;
      match.maneuver = templates_[frame_best].maneuver;
      match.best_score = match.frame_scores[frame_best];
    }
    match.reason = "weak_template_evidence";
    last_match_ = match;
    return match;
  }

  match.eligible = true;
  for (size_t i = 0; i < evidence_.size(); ++i) {
    evidence_[i] = config_.evidence_decay * evidence_[i] + match.frame_scores[i];
  }
  int best = 0;
  int second = evidence_.size() > 1 ? 1 : 0;
  if (evidence_[second] > evidence_[best]) {
    std::swap(best, second);
  }
  for (size_t i = 2; i < evidence_.size(); ++i) {
    if (evidence_[i] > evidence_[best]) {
      second = best;
      best = static_cast<int>(i);
    } else if (evidence_[i] > evidence_[second]) {
      second = static_cast<int>(i);
    }
  }
  if (best == consecutive_winner_) {
    ++consecutive_wins_;
  } else {
    consecutive_winner_ = best;
    consecutive_wins_ = 1;
  }

  match.best_index = best;
  match.best_id = templates_[best].id;
  match.maneuver = templates_[best].maneuver;
  match.best_score = evidence_[best];
  match.margin = evidence_[best] - evidence_[second];
  match.consecutive_wins = consecutive_wins_;
  match.evidence = evidence_;
  if (consecutive_wins_ >= config_.stable_frames &&
      match.best_score >= config_.stable_min_evidence &&
      match.margin >= config_.stable_min_margin) {
    stable_index_ = best;
    match.stable = true;
    match.reason = "stable";
  } else {
    match.reason = "accumulating";
  }
  last_match_ = match;
  return match;
}

}  // namespace track_perception_cpp
