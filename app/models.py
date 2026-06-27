"""Application state and data models.

State is split into four blobs for routing and debugging:
- TurnState: this turn's input and intermediate outputs (ephemeral).
- CaseState: problem category, slots, readiness, intent (session-scoped).
- UserProfile: culture, profile_notes, boundaries (long-term).
- SystemMeta: turn_id, timestamps, observability.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ----- Structured agent outputs (for confidence and selector) -----


class EmotionOutput(BaseModel):
    reflection: Optional[str] = None
    validation: Optional[str] = None
    gentle_reframe: Optional[str] = None
    confidence: float = 0.0


class CoachOutput(BaseModel):
    exercise_title: Optional[str] = None
    steps: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    warning_notes: Optional[str] = None
    confidence: float = 0.0


class GrowthOutput(BaseModel):
    goal: Optional[str] = None
    smart_breakdown: Optional[str] = None
    milestones: List[str] = Field(default_factory=list)
    obstacles: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class ClassificationOutput(BaseModel):
    problem_category: Optional[str] = None
    confidence: float = 0.0
    missing_slots: List[str] = Field(default_factory=list)


# ----- State blobs -----


class TurnState(BaseModel):
    """This turn's input and intermediate outputs. Ephemeral; reset each turn."""

    text: Optional[str] = None
    audio_path: Optional[str] = None
    active_speaker: str = "A"
    partner_id: Optional[str] = None
    recent_user_messages: List[str] = Field(default_factory=list)
    # Triage
    turn_type: Optional[str] = None  # smalltalk_only | intake_needed | venting_emotion | advice_coaching | growth_plan | knowledge_rag | high_risk_escalation | mixed
    user_intent: Optional[str] = None  # coach | growth | vent | mixed
    emotional_intensity: float = 0.0
    response_style: Optional[str] = None  # empathy_only | understanding | empathy_light_advice | full_advice
    needs_rag: bool = False
    # Risk
    risk_level: Optional[str] = None  # none | low | medium | high
    risk_type: Optional[str] = None  # self_harm | abuse | child_safety | violence | stalking | none
    risk_action: Optional[str] = None  # continue | ask_safety_question | provide_emergency_guidance | escalate_to_human
    # Classification / emotion (this turn)
    emotion: Optional[str] = None
    sentiment: Optional[str] = None
    # Scheduler flags (which agents to run this turn)
    run_emotion: bool = False
    run_coach: bool = False
    run_growth: bool = False
    run_rag: bool = False
    run_psychoeducation: bool = False
    run_pattern: bool = False
    # Agent outputs (structured where applicable)
    emotion_response: Optional[str] = None
    emotion_output: Optional[EmotionOutput] = None
    coach_response: Optional[str] = None
    coach_output: Optional[CoachOutput] = None
    growth_response: Optional[str] = None
    growth_output: Optional[GrowthOutput] = None
    psychoeducation_response: Optional[str] = None
    pattern_response: Optional[str] = None
    retrieved_info: Optional[str] = None
    cultural_note: Optional[str] = None
    phrasing_guidelines: Optional[str] = None
    # Adaptive turn routing
    turn_mode: Optional[str] = None  # safety_check | empathy_containment | clarification | intake_slot_fill | psychoeducation | communication_coaching | trust_repair | closeness_building | maintenance_review | progress_reflection
    turn_mode_reason: Optional[str] = None
    # Safety override
    safety_override_triggered: bool = False
    safety_flags: List[str] = Field(default_factory=list)
    # Safety gate
    allowed_actions: List[str] = Field(default_factory=list)
    must_ask: Optional[str] = None
    must_refuse: Optional[str] = None
    escalate: bool = False
    # Dialogue
    dialogue_action: Optional[str] = None  # ASK_ONE_QUESTION | RESPOND_AND_OPTIONAL_QUESTION | RESPOND_ONLY
    follow_up_question: Optional[str] = None
    # Selector bundle (input to composer)
    selected_empathy: Optional[str] = None
    selected_advice: Optional[str] = None
    selected_question: Optional[str] = None
    tone_guidance: Optional[str] = None
    detected_horsemen: List[str] = Field(default_factory=list)
    # Final
    final_response: Optional[str] = None
    smalltalk_response: Optional[str] = None


class TherapyPhaseState(BaseModel):
    """Tracks the current therapy phase and progress across the 5-phase program."""

    current_phase: int = 1  # 1-5
    session_id: Optional[str] = None
    turns_in_phase: int = 0
    total_turns: int = 0
    phase_goals: List[str] = Field(default_factory=list)
    therapy_approach: str = "integrative"
    milestones: Dict[str, bool] = Field(default_factory=dict)
    phase_notes: Optional[str] = None
    phase_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Adaptive phase control
    phase_confidence: float = 1.0
    phase_transition_decision: Optional[str] = None  # stay | advance | temporary_fallback | regress | review_needed
    phase_transition_reason: Optional[str] = None
    temporary_fallback: bool = False


class CaseState(BaseModel):
    """Problem category, slots, readiness, intent. Persists across turns in session."""

    problem_category: Optional[str] = None
    slots_filled: Dict[str, str] = Field(default_factory=dict)
    readiness_score: float = 0.0
    user_intent: Optional[str] = None  # coach | growth | vent | mixed
    intake_progress: Optional[str] = None
    plan: Optional[str] = None
    therapy_mode: str = "one_person"
    formulation_summary: Optional[str] = None
    strengths_summary: Optional[str] = None
    focus_areas: List[str] = Field(default_factory=list)
    conflict_pattern_assessment: Optional[str] = None
    classification_output: Optional[ClassificationOutput] = None
    intake_completed: bool = False
    questions_asked: int = 0
    # Adaptive routing metadata
    context_modifier: Optional[str] = None  # ordinary_conflict | repair_after_breach | high_escalation | possible_abuse | separation_or_breakup | one_partner_unavailable | individual_reflection_mode
    readiness_reason: Optional[str] = None
    soft_signals_detected: List[str] = Field(default_factory=list)
    milestones_completed: List[str] = Field(default_factory=list)
    coaching_eligible: bool = False
    coaching_eligibility_reason: Optional[str] = None


class UserProfile(BaseModel):
    """Stable facts: culture, preferences, boundaries. Long-term."""

    culture: Optional[str] = None
    gender: Optional[str] = None
    boundaries: Optional[str] = None
    profile_notes: Optional[str] = None
    prior_summaries: List[str] = Field(default_factory=list)
    profile: Dict[str, str] = Field(default_factory=dict)


class SystemMeta(BaseModel):
    """Observability: turn_id, timestamps, latency."""

    turn_id: Optional[str] = None
    timestamp: Optional[str] = None
    latency_ms: Optional[float] = None
    model_costs: Optional[Dict[str, Any]] = None


# ----- Top-level graph state -----


class AppState(BaseModel):
    """Graph state: composition of TurnState, CaseState, UserProfile, SystemMeta.

    Existing agents can still read flattened fields via the compatibility
    properties (e.g. state.text -> state.turn.text). New code should use
    state.turn, state.case, state.profile, state.meta.
    """

    turn: TurnState = Field(default_factory=TurnState)
    case: CaseState = Field(default_factory=CaseState)
    profile: UserProfile = Field(default_factory=UserProfile)
    meta: SystemMeta = Field(default_factory=SystemMeta)
    therapy: TherapyPhaseState = Field(default_factory=TherapyPhaseState)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

    # Backward-compatibility: flatten for existing nodes that read state.text etc.
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "AppState":
        """Build AppState from a flat dict (e.g. pipeline_input from UI)."""
        turn = TurnState(
            text=d.get("text"),
            audio_path=d.get("audio_path"),
            active_speaker=d.get("active_speaker") or "A",
            partner_id=d.get("partner_id"),
            recent_user_messages=d.get("recent_user_messages") or [],
            emotion=d.get("emotion"),
            sentiment=d.get("sentiment"),
            emotion_response=d.get("emotion_response"),
            coach_response=d.get("coach_response"),
            growth_response=d.get("growth_response"),
            retrieved_info=d.get("retrieved_info"),
            cultural_note=d.get("cultural_note"),
            final_response=d.get("final_response"),
            follow_up_question=d.get("follow_up_question"),
            smalltalk_response=d.get("smalltalk_response"),
            run_emotion=d.get("need_emotion", False),
            run_coach=d.get("need_coach", False),
            run_growth=d.get("need_growth", False),
            run_rag=d.get("needs_rag", False),
            run_psychoeducation=d.get("need_psychoeducation", False),
            run_pattern=d.get("need_pattern", False),
            turn_type=d.get("turn_type"),
            user_intent=d.get("user_intent"),
            emotional_intensity=float(d.get("emotional_intensity", 0) or 0),
            response_style=d.get("response_style"),
            needs_rag=d.get("needs_rag", False),
            risk_level=d.get("risk_level"),
            risk_type=d.get("risk_type"),
            risk_action=d.get("risk_action"),
            dialogue_action=d.get("dialogue_action"),
            detected_horsemen=d.get("detected_horsemen") or [],
            turn_mode=d.get("turn_mode"),
            turn_mode_reason=d.get("turn_mode_reason"),
            safety_override_triggered=d.get("safety_override_triggered", False),
            safety_flags=d.get("safety_flags") or [],
        )
        case = CaseState(
            problem_category=d.get("problem_category"),
            slots_filled=d.get("slots_filled") or {},
            readiness_score=float(d.get("readiness_score", 0) or 0),
            user_intent=d.get("user_need") or d.get("user_intent"),
            plan=d.get("plan"),
            therapy_mode=d.get("therapy_mode") or "one_person",
            formulation_summary=d.get("formulation_summary"),
            strengths_summary=d.get("strengths_summary"),
            focus_areas=d.get("focus_areas") or [],
            conflict_pattern_assessment=d.get("conflict_pattern_assessment"),
            intake_completed=d.get("intake_completed", False),
            questions_asked=int(d.get("questions_asked", 0) or 0),
            context_modifier=d.get("context_modifier"),
            readiness_reason=d.get("readiness_reason"),
            soft_signals_detected=d.get("soft_signals_detected") or [],
            milestones_completed=d.get("milestones_completed") or [],
            coaching_eligible=d.get("coaching_eligible", False),
            coaching_eligibility_reason=d.get("coaching_eligibility_reason"),
        )
        profile = UserProfile(
            culture=d.get("user_culture"),
            gender=d.get("gender"),
            profile_notes=d.get("profile_notes"),
            profile=d.get("profile") or {},
        )
        meta = SystemMeta(
            turn_id=d.get("turn_id"),
            timestamp=d.get("timestamp"),
            latency_ms=d.get("latency_ms"),
        )
        therapy = TherapyPhaseState(
            current_phase=int(d.get("current_phase", 1) or 1),
            session_id=d.get("session_id"),
            turns_in_phase=int(d.get("turns_in_phase", 0) or 0),
            total_turns=int(d.get("total_turns", 0) or 0),
            phase_goals=d.get("phase_goals") or [],
            therapy_approach=d.get("therapy_approach") or "integrative",
            milestones=d.get("milestones") or {},
            phase_notes=d.get("phase_notes"),
            phase_history=d.get("phase_history") or [],
            phase_confidence=float(d.get("phase_confidence", 1.0) or 1.0),
            phase_transition_decision=d.get("phase_transition_decision"),
            phase_transition_reason=d.get("phase_transition_reason"),
            temporary_fallback=d.get("temporary_fallback", False),
        )
        return cls(
            turn=turn,
            case=case,
            profile=profile,
            meta=meta,
            therapy=therapy,
            conversation_history=d.get("conversation_history") or [],
        )

    # Flattened read/write for backward compatibility
    @property
    def text(self) -> Optional[str]:
        return self.turn.text

    @text.setter
    def text(self, v: Optional[str]) -> None:
        self.turn.text = v

    @property
    def audio_path(self) -> Optional[str]:
        return self.turn.audio_path

    @audio_path.setter
    def audio_path(self, v: Optional[str]) -> None:
        self.turn.audio_path = v

    @property
    def recent_user_messages(self) -> List[str]:
        return self.turn.recent_user_messages

    @recent_user_messages.setter
    def recent_user_messages(self, v: List[str]) -> None:
        self.turn.recent_user_messages = v

    @property
    def emotion(self) -> Optional[str]:
        return self.turn.emotion

    @emotion.setter
    def emotion(self, v: Optional[str]) -> None:
        self.turn.emotion = v

    @property
    def sentiment(self) -> Optional[str]:
        return self.turn.sentiment

    @sentiment.setter
    def sentiment(self, v: Optional[str]) -> None:
        self.turn.sentiment = v

    @property
    def problem_category(self) -> Optional[str]:
        return self.case.problem_category

    @problem_category.setter
    def problem_category(self, v: Optional[str]) -> None:
        self.case.problem_category = v

    @property
    def plan(self) -> Optional[str]:
        return self.case.plan

    @plan.setter
    def plan(self, v: Optional[str]) -> None:
        self.case.plan = v

    @property
    def emotion_response(self) -> Optional[str]:
        return self.turn.emotion_response

    @emotion_response.setter
    def emotion_response(self, v: Optional[str]) -> None:
        self.turn.emotion_response = v

    @property
    def coach_response(self) -> Optional[str]:
        return self.turn.coach_response

    @coach_response.setter
    def coach_response(self, v: Optional[str]) -> None:
        self.turn.coach_response = v

    @property
    def growth_response(self) -> Optional[str]:
        return self.turn.growth_response

    @growth_response.setter
    def growth_response(self, v: Optional[str]) -> None:
        self.turn.growth_response = v

    @property
    def psychoeducation_response(self) -> Optional[str]:
        return self.turn.psychoeducation_response

    @psychoeducation_response.setter
    def psychoeducation_response(self, v: Optional[str]) -> None:
        self.turn.psychoeducation_response = v

    @property
    def pattern_response(self) -> Optional[str]:
        return self.turn.pattern_response

    @pattern_response.setter
    def pattern_response(self, v: Optional[str]) -> None:
        self.turn.pattern_response = v

    @property
    def retrieved_info(self) -> Optional[str]:
        return self.turn.retrieved_info

    @retrieved_info.setter
    def retrieved_info(self, v: Optional[str]) -> None:
        self.turn.retrieved_info = v

    @property
    def cultural_note(self) -> Optional[str]:
        return self.turn.cultural_note

    @cultural_note.setter
    def cultural_note(self, v: Optional[str]) -> None:
        self.turn.cultural_note = v

    @property
    def final_response(self) -> Optional[str]:
        return self.turn.final_response

    @final_response.setter
    def final_response(self, v: Optional[str]) -> None:
        self.turn.final_response = v

    @property
    def profile_notes(self) -> Optional[str]:
        return self.profile.profile_notes

    @profile_notes.setter
    def profile_notes(self, v: Optional[str]) -> None:
        self.profile.profile_notes = v

    @property
    def follow_up_question(self) -> Optional[str]:
        return self.turn.follow_up_question

    @follow_up_question.setter
    def follow_up_question(self, v: Optional[str]) -> None:
        self.turn.follow_up_question = v

    @property
    def smalltalk_response(self) -> Optional[str]:
        return self.turn.smalltalk_response

    @smalltalk_response.setter
    def smalltalk_response(self, v: Optional[str]) -> None:
        self.turn.smalltalk_response = v

    @property
    def user_culture(self) -> Optional[str]:
        return self.profile.culture

    @user_culture.setter
    def user_culture(self, v: Optional[str]) -> None:
        self.profile.culture = v

    @property
    def user_need(self) -> Optional[str]:
        return self.case.user_intent

    @user_need.setter
    def user_need(self, v: Optional[str]) -> None:
        self.case.user_intent = v

    @property
    def need_emotion(self) -> bool:
        return self.turn.run_emotion

    @need_emotion.setter
    def need_emotion(self, v: bool) -> None:
        self.turn.run_emotion = v

    @property
    def need_coach(self) -> bool:
        return self.turn.run_coach

    @need_coach.setter
    def need_coach(self, v: bool) -> None:
        self.turn.run_coach = v

    @property
    def need_growth(self) -> bool:
        return self.turn.run_growth

    @need_growth.setter
    def need_growth(self, v: bool) -> None:
        self.turn.run_growth = v

    @property
    def need_psychoeducation(self) -> bool:
        return self.turn.run_psychoeducation

    @need_psychoeducation.setter
    def need_psychoeducation(self, v: bool) -> None:
        self.turn.run_psychoeducation = v

    @property
    def need_pattern(self) -> bool:
        return self.turn.run_pattern

    @need_pattern.setter
    def need_pattern(self, v: bool) -> None:
        self.turn.run_pattern = v

    @property
    def intake_completed(self) -> bool:
        return self.case.intake_completed

    @intake_completed.setter
    def intake_completed(self, v: bool) -> None:
        self.case.intake_completed = v

    @property
    def questions_asked(self) -> int:
        return self.case.questions_asked

    @questions_asked.setter
    def questions_asked(self, v: int) -> None:
        self.case.questions_asked = v

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        d = super().model_dump(**kwargs)
        # Expose flattened keys for UI and persistence
        d["text"] = self.turn.text
        d["audio_path"] = self.turn.audio_path
        d["recent_user_messages"] = self.turn.recent_user_messages
        d["emotion"] = self.turn.emotion
        d["sentiment"] = self.turn.sentiment
        d["problem_category"] = self.case.problem_category
        d["plan"] = self.case.plan
        d["therapy_mode"] = self.case.therapy_mode
        d["formulation_summary"] = self.case.formulation_summary
        d["strengths_summary"] = self.case.strengths_summary
        d["focus_areas"] = self.case.focus_areas
        d["conflict_pattern_assessment"] = self.case.conflict_pattern_assessment
        d["emotion_response"] = self.turn.emotion_response
        d["coach_response"] = self.turn.coach_response
        d["growth_response"] = self.turn.growth_response
        d["retrieved_info"] = self.turn.retrieved_info
        d["cultural_note"] = self.turn.cultural_note
        d["final_response"] = self.turn.final_response
        d["profile_notes"] = self.profile.profile_notes
        d["follow_up_question"] = self.turn.follow_up_question
        d["smalltalk_response"] = self.turn.smalltalk_response
        d["active_speaker"] = self.turn.active_speaker
        d["partner_id"] = self.turn.partner_id
        d["user_culture"] = self.profile.culture
        d["user_need"] = self.case.user_intent
        d["turn_id"] = self.meta.turn_id
        d["response_style"] = self.turn.response_style
        d["need_emotion"] = self.turn.run_emotion
        d["need_coach"] = self.turn.run_coach
        d["need_growth"] = self.turn.run_growth
        d["needs_rag"] = self.turn.run_rag
        d["need_psychoeducation"] = self.turn.run_psychoeducation
        d["need_pattern"] = self.turn.run_pattern
        d["psychoeducation_response"] = self.turn.psychoeducation_response
        d["pattern_response"] = self.turn.pattern_response
        d["detected_horsemen"] = self.turn.detected_horsemen
        d["intake_completed"] = self.case.intake_completed
        d["questions_asked"] = self.case.questions_asked
        d["slots_filled"] = self.case.slots_filled
        d["readiness_score"] = self.case.readiness_score
        d["context_modifier"] = self.case.context_modifier
        d["readiness_reason"] = self.case.readiness_reason
        d["soft_signals_detected"] = self.case.soft_signals_detected
        d["milestones_completed"] = self.case.milestones_completed
        d["coaching_eligible"] = self.case.coaching_eligible
        d["coaching_eligibility_reason"] = self.case.coaching_eligibility_reason
        d["turn_mode"] = self.turn.turn_mode
        d["turn_mode_reason"] = self.turn.turn_mode_reason
        d["safety_override_triggered"] = self.turn.safety_override_triggered
        d["safety_flags"] = self.turn.safety_flags
        # Therapy phase state
        d["current_phase"] = self.therapy.current_phase
        d["session_id"] = self.therapy.session_id
        d["turns_in_phase"] = self.therapy.turns_in_phase
        d["total_turns"] = self.therapy.total_turns
        d["phase_goals"] = self.therapy.phase_goals
        d["therapy_approach"] = self.therapy.therapy_approach
        d["milestones"] = self.therapy.milestones
        d["phase_notes"] = self.therapy.phase_notes
        d["phase_history"] = self.therapy.phase_history
        d["phase_confidence"] = self.therapy.phase_confidence
        d["phase_transition_decision"] = self.therapy.phase_transition_decision
        d["phase_transition_reason"] = self.therapy.phase_transition_reason
        d["temporary_fallback"] = self.therapy.temporary_fallback
        return d
