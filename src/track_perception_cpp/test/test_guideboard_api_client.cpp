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

TEST(GuideboardApiClientTest, LowConfidenceUncertainDecisionIsRejected) {
  EXPECT_FALSE(GuideboardApiClient::shouldAcceptDecision(true, 0.74f, 0.75));
}

TEST(GuideboardApiClientTest, InvalidConfidenceIsRejectedForUncertainDecision) {
  EXPECT_FALSE(GuideboardApiClient::shouldAcceptDecision(
      true, std::numeric_limits<float>::quiet_NaN(), 0.75));
  EXPECT_FALSE(GuideboardApiClient::shouldAcceptDecision(true, 1.1f, 0.75));
}

TEST(GuideboardApiClientTest, ThresholdIsClampedToUnitInterval) {
  EXPECT_TRUE(GuideboardApiClient::shouldAcceptDecision(true, 0.0f, -1.0));
  EXPECT_FALSE(GuideboardApiClient::shouldAcceptDecision(true, 0.99f, 2.0));
  EXPECT_TRUE(GuideboardApiClient::shouldAcceptDecision(true, 1.0f, 2.0));
}

}  // namespace
}  // namespace track_perception_cpp
