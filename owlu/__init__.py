"""OWLU — Open World Label Updating Module.

Three-module architecture:
    discovery   — Candidate Discovery Module
    writer      — Ontology-Constrained Writer
    absorption  — Prototype Absorption Module

Backward-compatible re-exports from the old flat layout are provided below.
"""

# === common types (canonical location) ===
from .common.types import (  # noqa: F401
    OWLUConfig,
    CandidatePhrase,
    MatchResult,
    GateDecision,
    LabelInfo,
    ProtoLabelCluster,
    ValidationSample,
    Vector,
    Matrix,
    ClusterState,
)

# === Module 1: Candidate Discovery ===
from .discovery import CandidateDiscovery  # noqa: F401
from .discovery.gate import LtceGate  # noqa: F401
from .discovery.phrase_generator import LLMPhraseGenerator, LLMOutputError  # noqa: F401
from .discovery.matcher import SemanticMatcher  # noqa: F401

# === Module 2: Ontology-Constrained Writer ===
from .writer import OntologyWriter  # noqa: F401
from .writer.label_bank import LabelBank  # noqa: F401
from .writer.constraints import OntologyConstraintChecker, ConstraintViolation  # noqa: F401
from .writer.persistence import LabelBankStore  # noqa: F401

# === Shared encoder ===
from .common.encoder import BertEncoder, SentenceTransformerEncoder  # noqa: F401

# === Module 3: Prototype Absorption ===
from .absorption import PrototypeAbsorption  # noqa: F401
from .absorption.fast_sync import fast_sync, fast_sync_model  # noqa: F401
from .absorption.slow_sync import slow_sync  # noqa: F401
from .absorption.metrics import (  # noqa: F401
    normalize,
    cosine_similarity,
    score_document,
    infer_topk,
    infer_above_threshold,
    recalibrate_threshold,
    default_text_encoder,
    blend_and_normalize_torch,
    recalibrate_model_threshold,
)

