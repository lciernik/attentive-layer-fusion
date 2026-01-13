from enum import Enum


class Task(Enum):
    FEATURE_EXTRACTION = "feature_extraction"
    LINEAR_PROBE = "linear_probe"
    ATTENTIVE_PROBE = "attentive_probe"
    REP2REP = "rep2rep"
    SAE_TRAINING = "sae-training"
    MODEL_SIMILARITY = "model_similarity"

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]

    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"Invalid task: '{value}'. Valid options are: {cls.values()}")
