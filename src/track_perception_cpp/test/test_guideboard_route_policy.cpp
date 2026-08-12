#include <gtest/gtest.h>

#include "track_perception_cpp/guideboard_route_policy.hpp"

namespace track_perception_cpp {
namespace {

TEST(GuideboardRoutePolicyTest, FirstStraightMakesSecondRightWithoutRecognition) {
  GuideboardRoutePolicy policy;
  EXPECT_TRUE(policy.recognitionRequired());
  ASSERT_TRUE(policy.prepareRecognitionSuccess("straight"));
  EXPECT_EQ(policy.decisionSource(), "first_api");
  ASSERT_TRUE(policy.commitSignedEncounter());
  EXPECT_STREQ(policy.phaseName(), "WAIT_SECOND_KNOWN");
  EXPECT_EQ(policy.firstAction(), "straight");
  EXPECT_EQ(policy.pendingSecondAction(), "right");
  EXPECT_FALSE(policy.recognitionRequired());

  ASSERT_TRUE(policy.prepareKnownDecision());
  EXPECT_EQ(policy.preparedAction(), "right");
  EXPECT_EQ(policy.decisionSource(), "second_opposite");
  ASSERT_TRUE(policy.commitSignedEncounter());
  EXPECT_STREQ(policy.phaseName(), "DONE");
  EXPECT_EQ(policy.signedEncounterIndex(), 2);
}

TEST(GuideboardRoutePolicyTest, FirstRightMakesSecondStraightWithoutRecognition) {
  GuideboardRoutePolicy policy;
  ASSERT_TRUE(policy.prepareRecognitionSuccess("right"));
  ASSERT_TRUE(policy.commitSignedEncounter());
  ASSERT_TRUE(policy.prepareKnownDecision());
  EXPECT_EQ(policy.preparedAction(), "straight");
  ASSERT_TRUE(policy.commitSignedEncounter());
  EXPECT_STREQ(policy.phaseName(), "DONE");
}

TEST(GuideboardRoutePolicyTest, FirstFailureRequiresSecondRecoveryRecognition) {
  GuideboardRoutePolicy policy;
  ASSERT_TRUE(policy.prepareRecognitionFailure());
  EXPECT_EQ(policy.preparedAction(), "straight");
  EXPECT_EQ(policy.decisionSource(), "first_api_failure_straight");
  ASSERT_TRUE(policy.commitSignedEncounter());
  EXPECT_STREQ(policy.phaseName(), "WAIT_SECOND_RECOVERY");
  EXPECT_TRUE(policy.recognitionRequired());

  ASSERT_TRUE(policy.prepareRecognitionSuccess("right"));
  EXPECT_EQ(policy.decisionSource(), "second_recovery_api");
  ASSERT_TRUE(policy.commitSignedEncounter());
  EXPECT_STREQ(policy.phaseName(), "DONE");
}

TEST(GuideboardRoutePolicyTest, SecondRecoveryFailureFallsBackStraightAndFinishes) {
  GuideboardRoutePolicy policy;
  ASSERT_TRUE(policy.prepareRecognitionFailure());
  ASSERT_TRUE(policy.commitSignedEncounter());
  ASSERT_TRUE(policy.prepareRecognitionFailure());
  EXPECT_EQ(policy.preparedAction(), "straight");
  EXPECT_EQ(policy.decisionSource(), "second_recovery_failure_straight");
  ASSERT_TRUE(policy.commitSignedEncounter());
  EXPECT_STREQ(policy.phaseName(), "DONE");
}

TEST(GuideboardRoutePolicyTest, DetectionAloneNeverAdvancesRoutePhase) {
  GuideboardRoutePolicy policy;
  EXPECT_FALSE(policy.prepareKnownDecision());
  EXPECT_STREQ(policy.phaseName(), "WAIT_FIRST");
  EXPECT_EQ(policy.signedEncounterIndex(), 0);
  EXPECT_FALSE(policy.commitSignedEncounter());
  EXPECT_STREQ(policy.phaseName(), "WAIT_FIRST");
}

}  // namespace
}  // namespace track_perception_cpp
