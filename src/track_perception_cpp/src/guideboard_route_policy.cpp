#include "track_perception_cpp/guideboard_route_policy.hpp"

namespace track_perception_cpp {

void GuideboardRoutePolicy::reset() {
  phase_ = GuideboardRoutePhase::WaitFirst;
  signed_encounter_index_ = 0;
  first_action_.clear();
  pending_second_action_.clear();
  clearPreparedDecision();
}

bool GuideboardRoutePolicy::prepareKnownDecision() {
  if (prepared_decision_valid_) {
    return true;
  }
  if (phase_ == GuideboardRoutePhase::WaitSecondKnown &&
      validManeuver(pending_second_action_)) {
    prepared_decision_valid_ = true;
    prepared_recognition_success_ = true;
    prepared_action_ = pending_second_action_;
    decision_source_ = "second_opposite";
    return true;
  }
  if (phase_ == GuideboardRoutePhase::Done) {
    prepared_decision_valid_ = true;
    prepared_recognition_success_ = false;
    prepared_action_ = "straight";
    decision_source_ = "done_default";
    return true;
  }
  return false;
}

bool GuideboardRoutePolicy::recognitionRequired() const {
  return !prepared_decision_valid_ &&
         (phase_ == GuideboardRoutePhase::WaitFirst ||
          phase_ == GuideboardRoutePhase::WaitSecondRecovery);
}

bool GuideboardRoutePolicy::prepareRecognitionSuccess(const std::string& maneuver) {
  if (!recognitionRequired() || !validManeuver(maneuver)) {
    return false;
  }
  prepared_decision_valid_ = true;
  prepared_recognition_success_ = true;
  prepared_action_ = maneuver;
  decision_source_ = phase_ == GuideboardRoutePhase::WaitFirst
                         ? "first_api"
                         : "second_recovery_api";
  return true;
}

bool GuideboardRoutePolicy::prepareRecognitionFailure() {
  if (!recognitionRequired()) {
    return false;
  }
  prepared_decision_valid_ = true;
  prepared_recognition_success_ = false;
  prepared_action_ = "straight";
  decision_source_ = phase_ == GuideboardRoutePhase::WaitFirst
                         ? "first_api_failure_straight"
                         : "second_recovery_failure_straight";
  return true;
}

bool GuideboardRoutePolicy::commitSignedEncounter() {
  if (!prepared_decision_valid_ || signed_encounter_index_ >= 2) {
    return false;
  }

  if (phase_ == GuideboardRoutePhase::WaitFirst) {
    ++signed_encounter_index_;
    if (prepared_recognition_success_) {
      first_action_ = prepared_action_;
      pending_second_action_ = opposite(first_action_);
      phase_ = GuideboardRoutePhase::WaitSecondKnown;
    } else {
      pending_second_action_.clear();
      phase_ = GuideboardRoutePhase::WaitSecondRecovery;
    }
  } else if (phase_ == GuideboardRoutePhase::WaitSecondKnown ||
             phase_ == GuideboardRoutePhase::WaitSecondRecovery) {
    ++signed_encounter_index_;
    pending_second_action_.clear();
    phase_ = GuideboardRoutePhase::Done;
  } else {
    return false;
  }

  clearPreparedDecision();
  return true;
}

const char* GuideboardRoutePolicy::phaseName() const {
  switch (phase_) {
    case GuideboardRoutePhase::WaitFirst:
      return "WAIT_FIRST";
    case GuideboardRoutePhase::WaitSecondKnown:
      return "WAIT_SECOND_KNOWN";
    case GuideboardRoutePhase::WaitSecondRecovery:
      return "WAIT_SECOND_RECOVERY";
    case GuideboardRoutePhase::Done:
      return "DONE";
  }
  return "WAIT_FIRST";
}

std::string GuideboardRoutePolicy::opposite(const std::string& maneuver) {
  return maneuver == "right" ? "straight" : "right";
}

bool GuideboardRoutePolicy::validManeuver(const std::string& maneuver) {
  return maneuver == "straight" || maneuver == "right";
}

void GuideboardRoutePolicy::clearPreparedDecision() {
  prepared_decision_valid_ = false;
  prepared_recognition_success_ = false;
  prepared_action_.clear();
  decision_source_.clear();
}

}  // namespace track_perception_cpp
