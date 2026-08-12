#pragma once

#include <string>

namespace track_perception_cpp {

enum class GuideboardRoutePhase {
  WaitFirst,
  WaitSecondKnown,
  WaitSecondRecovery,
  Done,
};

class GuideboardRoutePolicy {
 public:
  void reset();

  // Called when a GuideBoard belongs to a newly armed branch encounter.
  // Returns true when an action is already available and OCR/API must be skipped.
  bool prepareKnownDecision();
  bool recognitionRequired() const;
  bool prepareRecognitionSuccess(const std::string& maneuver);
  bool prepareRecognitionFailure();

  // Route state advances only when LaneDecision actually locks a signed branch.
  bool commitSignedEncounter();

  GuideboardRoutePhase phase() const { return phase_; }
  const char* phaseName() const;
  int signedEncounterIndex() const { return signed_encounter_index_; }
  const std::string& firstAction() const { return first_action_; }
  const std::string& pendingSecondAction() const { return pending_second_action_; }
  bool hasPreparedDecision() const { return prepared_decision_valid_; }
  const std::string& preparedAction() const { return prepared_action_; }
  const std::string& decisionSource() const { return decision_source_; }

  static std::string opposite(const std::string& maneuver);

 private:
  static bool validManeuver(const std::string& maneuver);
  void clearPreparedDecision();

  GuideboardRoutePhase phase_{GuideboardRoutePhase::WaitFirst};
  int signed_encounter_index_{0};
  std::string first_action_;
  std::string pending_second_action_;
  bool prepared_decision_valid_{false};
  bool prepared_recognition_success_{false};
  std::string prepared_action_;
  std::string decision_source_;
};

}  // namespace track_perception_cpp
