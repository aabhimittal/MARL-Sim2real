"""Self-correction: recalibrate the GNN when sim2real drift is detected.

Two complementary mechanisms, both applied when a DriftEvent fires:

1. Baseline EMA update — the sim baseline mean for drifted edges is pulled
   toward the observed real-world mean, so the detector doesn't re-fire
   forever on the same (now known) reality gap.

2. GNN edge-bias fine-tuning — the DynamicGNN carries a per-edge calibration
   bias. We freeze the shared message-passing weights and take a few gradient
   steps on ONLY those biases against recent real observations. Path planning
   immediately reflects the corrected latencies; the shared network is not
   destabilised by a handful of local measurements.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from marl_sim2real.config import DriftConfig
from marl_sim2real.gnn.dynamic_gnn import DynamicGNN
from marl_sim2real.sim2real.drift_detector import DriftDetector, DriftEvent


@dataclasses.dataclass
class CorrectionRecord:
    tick: int
    edge_ids: list
    pre_error: float    # mean |predicted - real| on drifted edges before
    post_error: float   # ... and after recalibration
    wall_time_s: float


class SelfCorrection:
    def __init__(
        self,
        gnn: DynamicGNN,
        detector: DriftDetector,
        config: DriftConfig | None = None,
        log_path: str | Path | None = None,
    ):
        self.gnn = gnn
        self.detector = detector
        self.config = config or DriftConfig()
        self.log_path = Path(log_path) if log_path else None
        self.history: list[CorrectionRecord] = []

    def handle(self, event: DriftEvent, snapshots: list, real_targets: list) -> CorrectionRecord:
        """Recalibrate for a drift event.

        snapshots: recent (node_feats, edge_feats) graph states.
        real_targets: matching (E,) real-world latency arrays.
        """
        start = time.monotonic()
        edge_ids = event.edge_ids

        pre_error = self._edge_error(snapshots, real_targets, edge_ids)

        # 1. Pull the detector baseline toward reality (EMA) so the same gap
        #    doesn't re-trigger; widen std slightly to reflect new uncertainty.
        alpha = self.config.ema_alpha
        baseline = self.detector.baseline
        baseline.mean[edge_ids] = (
            (1 - alpha) * baseline.mean[edge_ids] + alpha * event.observed_mean
        )
        baseline.std[edge_ids] *= 1.0 + alpha

        # 2. Fine-tune only the per-edge calibration biases of the GNN.
        self.gnn.fit(
            snapshots,
            real_targets,
            epochs=max(1, self.config.recalib_steps // max(len(snapshots), 1)),
            lr=self.config.bias_lr,
            only_bias=True,
            edge_mask=edge_ids,
        )

        post_error = self._edge_error(snapshots, real_targets, edge_ids)
        self.detector.reset_edges(edge_ids)

        record = CorrectionRecord(
            tick=event.tick,
            edge_ids=[int(e) for e in edge_ids],
            pre_error=pre_error,
            post_error=post_error,
            wall_time_s=time.monotonic() - start,
        )
        self.history.append(record)
        self._log(record, event)
        return record

    # ------------------------------------------------------------- internals
    def _edge_error(self, snapshots: list, targets: list, edge_ids: np.ndarray) -> float:
        errors = []
        for (nf, ef), y in zip(snapshots, targets):
            pred = self.gnn.predict_latencies(nf, ef)
            errors.append(np.abs(pred[edge_ids] - y[edge_ids]).mean())
        return float(np.mean(errors)) if errors else float("nan")

    def _log(self, record: CorrectionRecord, event: DriftEvent) -> None:
        line = (
            f"[self-correction] {event.summary()} | "
            f"edge error {record.pre_error:.3f}s -> {record.post_error:.3f}s "
            f"in {record.wall_time_s:.2f}s"
        )
        print(line)
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(dataclasses.asdict(record)) + "\n")
