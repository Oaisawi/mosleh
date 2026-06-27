"""Configuration and constants for the counseling assistant."""
import os
from typing import Optional


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a setting from env first, then Streamlit secrets when available."""
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            secret_value = st.secrets[name]
            return str(secret_value) if secret_value is not None else default
    except Exception:
        pass
    return default


def _int_setting(name: str, default: int) -> int:
    value = get_setting(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# LLM provider
MODEL_PROVIDER = (get_setting("MODEL_PROVIDER", "openai") or "openai").lower()
GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
OPENAI_API_KEY = get_setting("OPENAI_API_KEY")

# Model names
OPENAI_MODEL_NAME = get_setting("OPENAI_MODEL_NAME", "gpt-5-nano")
GEMINI_MODEL_NAME = get_setting("GEMINI_MODEL_NAME", "gemini-2.5-flash")
MODEL_NAME = get_setting("MODEL_NAME") or (
    GEMINI_MODEL_NAME if MODEL_PROVIDER == "gemini" else OPENAI_MODEL_NAME
)

# Embeddings
EMBEDDING_MODEL = get_setting("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIM = _int_setting("EMBEDDING_DIM", 1536)
EMBED_DIM = 256

# Optional Qdrant-backed RAG. In "auto", Qdrant is used only when fully configured.
QDRANT_URL = get_setting("QDRANT_URL")
QDRANT_API_KEY = get_setting("QDRANT_API_KEY")
QDRANT_COLLECTION = get_setting("QDRANT_COLLECTION")
RAG_TOP_K = _int_setting("RAG_TOP_K", 3)
RAG_BACKEND = (get_setting("RAG_BACKEND", "auto") or "auto").lower()

# ---------------------------------------------------------------------------
# Therapy Phases — 5-phase couples therapy program
# ---------------------------------------------------------------------------

THERAPY_PHASES = {
    1: {
        "name_ar": "التقييم وبناء العلاقة العلاجية",
        "name_en": "Assessment & Building Therapeutic Relationship",
        "objectives": [
            "Getting to know the couple and building safety",
            "Assessing the nature of the problem and relationship history",
            "Setting each partner's expectations",
            "Establishing shared therapy goals",
        ],
        "tools": [
            "Counseling interview",
            "Marital satisfaction assessment",
            "Interaction observation",
        ],
        "milestones": [
            "safety_established",
            "problem_assessed",
            "expectations_set",
            "goals_defined",
        ],
        "min_turns": 3,
        "description": (
            "The opening phase focuses on building a safe therapeutic space. "
            "The counselor learns about the couple's relationship history, "
            "identifies the core issues, captures each partner's expectations, "
            "and co-creates shared therapy goals."
        ),
    },
    2: {
        "name_ar": "فهم الذات والطرف الآخر",
        "name_en": "Understanding Self & Partner",
        "objectives": [
            "Exploring personality patterns and their effect on marriage",
            "Understanding cognitive and communicative differences",
            "Discovering unmet emotional needs",
        ],
        "tools": [
            "Self-awareness exercises",
            "Needs mapping",
            "Homework assignments",
        ],
        "milestones": [
            "personality_explored",
            "differences_discussed",
            "unmet_needs_identified",
        ],
        "min_turns": 3,
        "description": (
            "This phase helps each partner understand their own personality, "
            "communication style, and emotional needs — and how these interact "
            "with their partner's. Awareness is the foundation for change."
        ),
    },
    3: {
        "name_ar": "مهارات التواصل وإدارة الخلاف",
        "name_en": "Communication Skills & Conflict Management",
        "objectives": [
            "Developing active listening skills",
            "Expressing feelings without blame",
            "Managing anger constructively",
            "Joint problem solving and decision making",
        ],
        "tools": [
            "Role playing",
            "Reframing exercises",
            "Practical communication training",
        ],
        "milestones": [
            "active_listening_practiced",
            "nonblame_expression_learned",
            "anger_management_discussed",
            "problem_solving_practiced",
        ],
        "min_turns": 4,
        "description": (
            "The skills-building phase teaches concrete communication and "
            "conflict-resolution techniques. Couples learn to listen actively, "
            "express needs without blame, manage anger, and make decisions together."
        ),
    },
    4: {
        "name_ar": "بناء الثقة والقرب العاطفي",
        "name_en": "Building Trust & Emotional Closeness",
        "objectives": [
            "Understanding causes of trust erosion",
            "Learning proper apology and forgiveness",
            "Enhancing appreciation and caring",
            "Reviving the emotional connection",
        ],
        "tools": [
            "Appreciation letters",
            "Dedicated couple time",
            "Positive rituals",
        ],
        "milestones": [
            "trust_erosion_understood",
            "apology_forgiveness_practiced",
            "appreciation_enhanced",
            "emotional_connection_revived",
        ],
        "min_turns": 3,
        "description": (
            "This phase rebuilds the emotional foundation — addressing broken "
            "trust, practicing genuine apology and forgiveness, and creating "
            "new rituals of appreciation and closeness."
        ),
    },
    5: {
        "name_ar": "التثبيت والوقاية",
        "name_en": "Stabilization & Prevention",
        "objectives": [
            "Reviewing what has been learned",
            "Creating a plan for future conflicts",
            "Strengthening independence while maintaining connection",
            "Final progress assessment",
        ],
        "tools": [
            "Progress review",
            "Conflict prevention plan",
            "Relapse prevention strategies",
        ],
        "milestones": [
            "learning_reviewed",
            "future_plan_created",
            "independence_with_connection",
            "final_assessment_done",
        ],
        "min_turns": 2,
        "description": (
            "The closing phase consolidates gains, builds a concrete plan for "
            "handling future disagreements, balances autonomy with togetherness, "
            "and conducts a final assessment of progress."
        ),
    },
}

# ---------------------------------------------------------------------------
# Phase response-mode policies — preferred / allowed / blocked per phase
# Used by turn_router and specialist_orchestrator instead of hard locks.
# ---------------------------------------------------------------------------

PHASE_POLICIES = {
    1: {
        "preferred": {"empathy_containment", "clarification", "intake_slot_fill", "safety_check"},
        "allowed_limited": {"psychoeducation", "progress_reflection"},
        "blocked": {"communication_coaching", "trust_repair", "closeness_building", "maintenance_review"},
    },
    2: {
        "preferred": {"psychoeducation", "empathy_containment", "clarification", "progress_reflection"},
        "allowed_limited": {"communication_coaching", "intake_slot_fill"},
        "blocked": {"trust_repair", "closeness_building", "maintenance_review"},
    },
    3: {
        "preferred": {"communication_coaching", "psychoeducation", "empathy_containment"},
        "allowed_limited": {"trust_repair", "progress_reflection", "clarification"},
        "blocked": {"maintenance_review"},
    },
    4: {
        "preferred": {"trust_repair", "closeness_building", "empathy_containment", "communication_coaching"},
        "allowed_limited": {"maintenance_review", "psychoeducation", "progress_reflection"},
        "blocked": set(),
    },
    5: {
        "preferred": {"maintenance_review", "progress_reflection", "communication_coaching"},
        "allowed_limited": {"empathy_containment", "trust_repair", "closeness_building", "psychoeducation"},
        "blocked": set(),
    },
}

# Soft readiness signals used for weighted phase-progression evidence.
# Every entry here MUST have a matching detector in phase_manager._detect_soft_signals;
# the list length is the denominator for soft_pct, so unreachable entries suppress scores.
SOFT_READINESS_SIGNALS = [
    "emotional_regulation",
    "describes_feelings_clearly",
    "reflects_on_own_role",
    "considers_partner_perspective",
    "open_to_practical_tools",
    "reports_prior_advice_useful",
    "willing_to_try_new_approach",
]

# Context modifier detection cues
CONTEXT_MODIFIER_CUES = {
    "repair_after_breach": {
        "betrayed", "cheated", "affair", "lied to me", "broke my trust",
        "found out", "caught", "unfaithful", "infidelity", "betrayal",
    },
    "separation_or_breakup": {
        "divorce", "break up", "breaking up", "separation", "separate",
        "it's over", "ending it", "leaving", "left me", "moved out",
        "want out", "can't do this anymore",
    },
    "high_escalation": {
        "screaming", "yelling", "throwing things", "slammed", "punched wall",
        "can't stop fighting", "explosive", "blew up", "lost it",
    },
    "one_partner_unavailable": {
        "won't come", "refuses therapy", "partner won't", "doing this alone",
        "only one here", "they don't care", "won't talk",
    },
}

# Coercive control and abuse indicator cues (broader than crisis keywords)
COERCIVE_CONTROL_CUES = {
    "controls my", "controls me", "won't let me", "not allowed to",
    "checks my phone", "monitors me", "isolates me", "cut me off from",
    "takes my money", "financial control", "threatens to leave with the kids",
    "says i'm crazy", "gaslighting", "manipulates", "makes me feel crazy",
    "punishes me", "silent treatment for days", "withholds",
    "tells me what to wear", "can't see my friends", "can't see my family",
    "intimidates", "scares me", "walks on eggshells",
}

THERAPY_APPROACHES = {
    "integrative": {
        "name_ar": "الإرشاد الزواجي التكاملي",
        "name_en": "Integrative Marriage Counseling",
        "description": "Combines multiple therapeutic modalities for a holistic approach.",
    },
    "cbt": {
        "name_ar": "العلاج المعرفي السلوكي الزواجي",
        "name_en": "Cognitive Behavioral Therapy for Couples",
        "description": "Focuses on identifying and changing negative thought patterns and behaviors.",
    },
    "eft": {
        "name_ar": "العلاج القائم على المشاعر",
        "name_en": "Emotion-Focused Therapy (EFT)",
        "description": "Centers on emotional bonds and attachment needs between partners.",
    },
    "value_based": {
        "name_ar": "الإرشاد القيمي",
        "name_en": "Value-Based Counseling",
        "description": "Aligns therapy with the couple's shared values and beliefs.",
    },
}

THERAPY_OUTCOMES = [
    "Noticeable improvement in communication",
    "Reduced conflict intensity",
    "Clearer roles and expectations",
    "Greater sense of safety and support",
    "Increased marital satisfaction and stability",
]

# Conversation cues and prompts
GREETINGS = {"hello", "hi", "hey", "salam", "good morning", "good evening"}
THERAPY_MODES = {"one_person", "two_partner"}
ACTION_CUES = {"what should", "how do", "advice", "help me", "need help", "suggest", "tips"}
PLAN_CUES = {"plan", "goal", "improve", "long term", "long-term", "future", "next step", "next steps"}

# Understanding / "why" cues — trigger psychoeducation agent
# Broad patterns to catch natural phrasing like "why do you think she...", "what made him..."
UNDERSTANDING_CUES = {
    # "why" variants — covers "why do we", "why do you think", "why did she", "why is he", etc.
    "why do", "why did", "why does", "why is", "why are", "why isn't", "why doesn't",
    "why won't", "why would", "why has", "why hasn't",
    # "what" variants — covers "what made her", "what caused", "what's going on"
    "what made", "what caused", "what causes", "what does it mean",
    "what's going on", "what is going on", "what's behind", "what is behind",
    "what happened", "what changed", "what went wrong", "what shifted",
    # Direct asks
    "explain", "how come", "reason", "understand why",
    # Common natural phrasings
    "do you think she", "do you think he", "do you think we",
    "what do you think", "any idea why", "can you explain",
}

# Pattern / cycle cues — trigger pattern/cycle agent
PATTERN_CUES = {
    "we keep", "same fight", "same argument", "cycle", "pattern",
    "always end up", "every time", "keeps happening", "stuck in",
    "we always", "going in circles", "never changes",
    "over and over", "again and again", "repeating",
}

FOUR_HORSEMEN_CUES = {
    "criticism": {"you always", "you never", "what is wrong with you", "your fault", "blame you"},
    "contempt": {"disgusting", "pathetic", "stupid", "worthless", "eye roll", "mocking"},
    "defensiveness": {"not my fault", "i did nothing wrong", "you started it", "why are you attacking me"},
    "stonewalling": {"shut down", "silent treatment", "walk away", "stop talking", "withdraw completely"},
}

# Category-based modality selection: which agents to run per problem category
# True = run by default for that category; can be overridden by turn_type
CATEGORY_MODALITIES = {
    "Communication":            {"emotion": True,  "coach": True,  "growth": False, "psychoeducation": True,  "pattern": True},
    "Emotional Distance":       {"emotion": True,  "coach": True,  "growth": True,  "psychoeducation": True,  "pattern": True},
    "Trust Issues":             {"emotion": True,  "coach": True,  "growth": True,  "psychoeducation": True,  "pattern": False},
    "Financial Stress":         {"emotion": True,  "coach": True,  "growth": True,  "psychoeducation": False, "pattern": False},
    "Child Related Conflicts":  {"emotion": True,  "coach": True,  "growth": True,  "psychoeducation": True,  "pattern": False},
    "Family Interference":      {"emotion": True,  "coach": True,  "growth": False, "psychoeducation": True,  "pattern": False},
    "Intimacy & Affection":     {"emotion": True,  "coach": True,  "growth": True,  "psychoeducation": True,  "pattern": True},
    "Cultural/Value Differences": {"emotion": True, "coach": True, "growth": True,  "psychoeducation": True,  "pattern": False},
    "Other":                    {"emotion": True,  "coach": True,  "growth": False, "psychoeducation": False, "pattern": False},
}
# Risk keywords for triage and risk guard (subset for high_risk_escalation)
RISK_KEYWORDS_HIGH = {
    "suicide", "kill myself", "end my life", "self-harm", "hurt myself",
    "abuse", "hit me", "hit her", "hit him", "violent", "stalk", "stalking",
    "child abuse", "hurt the kids", "danger", "unsafe", "scared for my life",
}
RISK_KEYWORDS_MEDIUM = {
    "depressed", "hopeless", "can't go on", "leave me", "threaten",
    "angry", "rage", "out of control",
}
INTAKE_QUESTIONS = [
    "What is the main issue between you and your partner right now?",
    "How long has this been going on?",
    "What have you tried so far to address it?",
    "How is this affecting you or your relationship day to day?",
    "Would you prefer practical steps right now, or a longer-term growth plan?",
]

# Slot-based intake: generic slots and per-category required slots
SLOTS_GENERIC = [
    "situation_summary",
    "who_involved",
    "timeframe",
    "what_tried",
    "desired_outcome",
    "constraints",
]
SLOTS_THERAPY_OPTIONAL = [
    "relationship_length",
    "how_met",
    "major_transitions",
    "conflict_triggers",
    "how_arguments_end",
    "repair_attempts",
    "relationship_strengths",
    "success_criteria",
    "partner_perspective",
]
SLOT_QUESTIONS = {
    "situation_summary": "What is the main issue between you and your partner right now?",
    "who_involved": "Who is involved (e.g. you and your partner, children, in-laws)?",
    "timeframe": "How long has this been going on?",
    "what_tried": "What have you tried so far to address it?",
    "desired_outcome": "What would you like to achieve or change?",
    "constraints": "Is there anything we should keep in mind (e.g. privacy, safety, children)?",
    "relationship_length": "How long have you been together?",
    "how_met": "How did your relationship begin, and what was strong about it early on?",
    "major_transitions": "What major transitions affected your relationship (e.g. marriage, parenthood, relocation)?",
    "conflict_triggers": "What topics or moments usually trigger conflict between you?",
    "how_arguments_end": "How do arguments usually end between you two?",
    "repair_attempts": "After conflict, how do you usually try to repair or reconnect?",
    "relationship_strengths": "What still works well between you, even when things are hard?",
    "success_criteria": "How will you know therapy is helping?",
    "partner_perspective": "How do you think your partner sees this situation?",
}
# Per-category required slots (default: all generic)
SLOTS_REQUIRED_BY_CATEGORY = {
    "Communication": SLOTS_GENERIC,
    "Financial Stress": SLOTS_GENERIC,
    "Child Related Conflicts": SLOTS_GENERIC,
    "Emotional Distance": SLOTS_GENERIC,
    "Family Interference": SLOTS_GENERIC,
    "Intimacy & Affection": SLOTS_GENERIC,
    "Trust Issues": SLOTS_GENERIC,
    "Cultural/Value Differences": SLOTS_GENERIC,
    "Other": SLOTS_GENERIC,
}
