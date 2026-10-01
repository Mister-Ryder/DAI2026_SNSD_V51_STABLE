"""Strict-zero OneShot communication runner v4.3.

After the sole real counterproposal at step 1, OneShotComm keeps the true
source pressure for provenance but applies the registered ``zero`` message
intervention.  Consequently proposal generation and TriGraph inference both
receive five exactly-zero dynamic message channels.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from .base_protocol import ExperimentResult
from .recurrent_protocol import RECURRENT_METHODS, RecurrentNegotiationConfig
from .fair_relay_protocol import FairRelayProtocol


class AnchoredRelayProtocol(FairRelayProtocol):
    def __init__(self, config: RecurrentNegotiationConfig) -> None:
        super().__init__(config)
        self._force_zero_intervention = False

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-recurrent-learned-run-v4.3",
                "one_shot_post_step1_intervention": "strict-zero-five-channels",
                "one_shot_zero_preserves_real_source_provenance": True,
            }
        )
        return fields

    def _pressure_for_step(self, step: int, pressure_maps):
        self._force_zero_intervention = False
        if int(step) == 0:
            return None, "none"
        if self.config.one_shot_comm and int(step) >= 2:
            # Preserve the genuine identified source.  Zeroing is performed by
            # the registered intervention inside _query_wave, after message
            # construction, so uncertainty and every other channel become 0.
            self._force_zero_intervention = True
            return pressure_maps, "zero"
        return super()._pressure_for_step(step, pressure_maps)

    def _query_wave(self, *args: Any, **kwargs: Any):
        if not self._force_zero_intervention:
            return super()._query_wave(*args, **kwargs)
        original = self.behavior
        self.behavior = replace(original, wave_b_intervention="zero")
        try:
            return super()._query_wave(*args, **kwargs)
        finally:
            self.behavior = original
            self._force_zero_intervention = False


def run_anchored_relay_protocol(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    smoke: bool = False,
    oracle: bool = False,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    allow_untrained_smoke: bool = False,
) -> ExperimentResult:
    one_shot = method == "OneShotComm"
    base_method = "Full" if one_shot else method
    if method not in RECURRENT_METHODS:
        raise ValueError("unknown recurrent method: {}".format(method))
    return AnchoredRelayProtocol(
        RecurrentNegotiationConfig(
            data_path=data_path,
            output_dir=output_dir,
            method=base_method,
            seed=seed,
            smoke=smoke,
            oracle=oracle,
            checkpoint=checkpoint,
            device=device,
            allow_untrained_smoke=allow_untrained_smoke,
            one_shot_comm=one_shot,
        )
    ).run()


__all__ = [
    "RECURRENT_METHODS",
    "RecurrentNegotiationConfig",
    "AnchoredRelayProtocol",
    "run_anchored_relay_protocol",
]
