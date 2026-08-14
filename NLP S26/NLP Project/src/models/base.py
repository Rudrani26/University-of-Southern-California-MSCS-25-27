from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """
    Abstract base class that every model backend must implement.

    predict() must always return a dict with exactly these keys:
        {
            "prediction":    str | None,   # single char: A, B, C, or D; None if unparseable
            "input_tokens":  int,
            "output_tokens": int,
            "latency_ms":    float,
        }

    predict() must NEVER raise — return prediction=None on any parse failure.
    """

    @abstractmethod
    def predict(self, prompt: str) -> dict:
        """
        Run inference on a single prompt and return a standardised result dict.

        Args:
            prompt: The full prompt string to send to the model.

        Returns:
            dict with keys: prediction, input_tokens, output_tokens, latency_ms.
            prediction is one of {A, B, C, D} or None if output is unparseable.
        """
        ...
