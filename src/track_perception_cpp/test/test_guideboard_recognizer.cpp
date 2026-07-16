#include <gtest/gtest.h>

#include "track_perception_cpp/guideboard_recognizer.hpp"

namespace track_perception_cpp {
namespace {

std::vector<GuideboardTemplateSpec> templates() {
  return {
      {"rough_right", "右道真的有点崎岖，但是直道真的走不了", "right"},
      {"irony_straight", "右道比直道好走多了？才怪！", "straight"},
      {"phone_straight", "他们来电话了，说不让我方向盘往右打！", "straight"},
      {"shortcut_right", "我就说三个字：抄近道", "right"},
      {"right_dead_end_straight", "右道是一条不归路！", "straight"},
      {"left_rough_right_flat", "左侧道路崎岖，右侧一马平川", "right"},
  };
}

GuideboardRecognizer makeRecognizer() {
  GuideboardRecognizer recognizer;
  std::string error;
  EXPECT_TRUE(recognizer.configure(templates(), GuideboardRecognizerConfig{}, &error)) << error;
  return recognizer;
}

TEST(GuideboardRecognizerTest, NormalizesUtf8AndPunctuation) {
  EXPECT_EQ(GuideboardRecognizer::normalizeUtf8(" 我就说三个字：抄近道！|\n"),
            "我就说三个字抄近道");
  EXPECT_EQ(GuideboardRecognizer::normalizeUtf8("ＡＢＣ：１２３"), "ABC123");
}

TEST(GuideboardRecognizerTest, ComputesLcsByUnicodeCodepoint) {
  EXPECT_FLOAT_EQ(GuideboardRecognizer::normalizedLcsRatio("右道！", "右道"), 1.0f);
  EXPECT_NEAR(GuideboardRecognizer::normalizedLcsRatio("右弯道", "右直道"), 2.0f / 3.0f,
              1e-6f);
  EXPECT_FLOAT_EQ(GuideboardRecognizer::normalizedLcsRatio("", "右道"), 0.0f);
}

TEST(GuideboardRecognizerTest, ExactTemplateHasUniqueCharacterBigrams) {
  auto recognizer = makeRecognizer();
  const auto match = recognizer.update(templates()[2].text, 0.70f);
  ASSERT_EQ(match.unique_bigram_hits.size(), templates().size());
  EXPECT_GE(match.unique_bigram_hits[2], 2);
  EXPECT_TRUE(match.eligible);
  EXPECT_EQ(match.best_id, "phone_straight");
}

TEST(GuideboardRecognizerTest, ClassifiesAllExactTemplatesAfterTwoFrames) {
  for (const auto& spec : templates()) {
    auto recognizer = makeRecognizer();
    auto first = recognizer.update(spec.text, 0.60f);
    EXPECT_TRUE(first.eligible) << spec.id;
    EXPECT_FALSE(first.stable) << spec.id;
    auto second = recognizer.update(spec.text, 0.60f);
    EXPECT_TRUE(second.stable) << spec.id;
    EXPECT_EQ(second.best_id, spec.id);
    EXPECT_EQ(second.maneuver, spec.maneuver);
  }
}

TEST(GuideboardRecognizerTest, ToleratesObservedOcrErrors) {
  struct Case {
    std::string text;
    std::string expected;
  };
  const std::vector<Case> cases = {
      {"但是真及的走不了", "rough_right"},
      {"右道真的有点崎邮，想是直道真的走不了。", "rough_right"},
      {"右道比直道好重条了？才怪！", "irony_straight"},
      {"他们束电遇了，说不让展方向进往右打！", "phone_straight"},
      {"我就说三个字：抄近道", "shortcut_right"},
      {"右道是一条无归路", "right_dead_end_straight"},
      {"方则是一条左归路右调是一张不归路", "right_dead_end_straight"},
      {"在现是一条无归降占调是一族不归始", "right_dead_end_straight"},
      {"左侧道路崎岖，右侧一马平川", "left_rough_right_flat"},
      {"左侧道路低岖，右侧一马平川", "left_rough_right_flat"},
      {"左创道路崎柜，右一马平川", "left_rough_right_flat"},
      {"定测道路情恒,右测一马平川", "left_rough_right_flat"},
  };
  for (const auto& test : cases) {
    auto recognizer = makeRecognizer();
    recognizer.update(test.text, 0.55f);
    auto result = recognizer.update(test.text, 0.55f);
    EXPECT_TRUE(result.stable) << test.text;
    EXPECT_EQ(result.best_id, test.expected) << test.text;
  }
}

TEST(GuideboardRecognizerTest, CommonOrWeakTextRemainsUnknown) {
  auto recognizer = makeRecognizer();
  for (int i = 0; i < 5; ++i) {
    auto result = recognizer.update("右道直道", 0.80f);
    EXPECT_FALSE(result.eligible);
    EXPECT_FALSE(result.stable);
    EXPECT_EQ(result.reason, "weak_template_evidence");
  }
  auto low_score = recognizer.update("我就说三个字抄近道", 0.20f);
  EXPECT_FALSE(low_score.eligible);
  EXPECT_EQ(low_score.reason, "low_model_score");
}

TEST(GuideboardRecognizerTest, ResetClearsStableEvidence) {
  auto recognizer = makeRecognizer();
  recognizer.update(templates()[0].text, 0.70f);
  ASSERT_TRUE(recognizer.update(templates()[0].text, 0.70f).stable);
  recognizer.reset();
  const auto& reset = recognizer.lastMatch();
  EXPECT_FALSE(reset.stable);
  for (float evidence : reset.evidence) {
    EXPECT_FLOAT_EQ(evidence, 0.0f);
  }
  EXPECT_FALSE(recognizer.update("右道直道", 0.70f).stable);
}

TEST(GuideboardRecognizerTest, RejectsInvalidConfiguration) {
  auto bad = templates();
  bad[0].maneuver = "left";
  GuideboardRecognizer recognizer;
  std::string error;
  EXPECT_FALSE(recognizer.configure(bad, GuideboardRecognizerConfig{}, &error));
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace track_perception_cpp
