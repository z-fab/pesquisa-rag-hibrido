from enum import Enum


class AblationMode(str, Enum):
    """Ablation study configurations for the agent pipeline."""

    FULL = "full"
    NO_VERIFIER = "no-verifier"
    NO_SYNTHESIZER = "no-synthesizer"
    POC = "poc"
