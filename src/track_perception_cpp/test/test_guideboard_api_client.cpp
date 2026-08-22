#include <gtest/gtest.h>

#include <limits>

#include "track_perception_cpp/guideboard_api_client.hpp"

namespace track_perception_cpp {
namespace {

TEST(GuideboardApiClientTest, CertainDecisionIsAlwaysAccepted) {
  EXPECT_TRUE(GuideboardApiClient::shouldAcceptDecision(false, 0.0f, 0.60));
}

TEST(GuideboardApiClientTest, HighConfidenceUncertainDecisionIsAccepted) {
  EXPECT_TRUE(GuideboardApiClient::shouldAcceptDecision(true, 0.75f, 0.75));
  EXPECT_TRUE(GuideboardApiClient::shouldAcceptDecision(true, 0.85f, 0.75));
}

TEST(GuideboardApiClientTest, LowConfidenceUncertainDecisionIsAccepted) {
  EXPECT_TRUE(GuideboardApiClient::shouldAcceptDecision(true, 0.01f, 0.75));
}

TEST(GuideboardApiClientTest, InvalidConfidenceIsRejected) {
  EXPECT_FALSE(GuideboardApiClient::shouldAcceptDecision(
      true, std::numeric_limits<float>::quiet_NaN(), 0.75));
  EXPECT_FALSE(GuideboardApiClient::shouldAcceptDecision(true, 1.1f, 0.75));
}

TEST(GuideboardApiClientTest, ThresholdDoesNotControlAcceptance) {
  EXPECT_TRUE(GuideboardApiClient::shouldAcceptDecision(true, 0.0f, -1.0));
  EXPECT_TRUE(GuideboardApiClient::shouldAcceptDecision(true, 0.99f, 2.0));
  EXPECT_TRUE(GuideboardApiClient::shouldAcceptDecision(true, 1.0f, 2.0));
}

TEST(GuideboardApiClientTest, SelectsBestDiverseSamplesAndRestoresTimeOrder) {
  const std::vector<GuideboardApiSample> samples{
      {"左边通行", 0.30f}, {"左边通行", 0.90f}, {"右边通行", 0.80f},
      {"", 1.00f}, {"直行", 0.70f}, {"右边通行", 0.20f}};
  const auto selected =
      GuideboardApiClient::selectDiverseSamples(samples, 2, 0.95);
  ASSERT_EQ(selected.size(), 2U);
  EXPECT_EQ(selected[0].text, "左边通行");
  EXPECT_FLOAT_EQ(selected[0].score, 0.90f);
  EXPECT_EQ(selected[1].text, "右边通行");
}

TEST(GuideboardApiClientTest, SampleSelectionHonorsMaximumWithoutCountGate) {
  const std::vector<GuideboardApiSample> one{{"只有一条文字", 0.10f}};
  const auto selected = GuideboardApiClient::selectDiverseSamples(one, 8, 0.70);
  ASSERT_EQ(selected.size(), 1U);
  EXPECT_EQ(selected.front().text, "只有一条文字");
}

}  // namespace
}  // namespace track_perception_cpp
