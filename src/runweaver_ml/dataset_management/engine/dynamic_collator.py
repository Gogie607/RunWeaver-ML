
import torch
import numpy as np


class DynamicCollator:

    def __init__(
        self,
        modalities,
    ):

        self.modalities = modalities

    # -------------------------------------------------

    def __call__(self, batch):

        out = {
            "sample_id": [
                x["sample_id"]
                for x in batch
            ],
            "__shard__": [
                x["__shard__"]
                for x in batch
            ],
            "domain": [
                x.get("domain")
                for x in batch
            ]
        }

        for modality in self.modalities:

            vals = [
                sample.get(modality)
                for sample in batch
            ]

            if all(v is None for v in vals):
                continue

            out[modality] = self._assemble(
                vals,
                modality=modality,
            )

        return out

    # -------------------------------------------------

    def _assemble(self, vals, modality):

        if any(v is None for v in vals):
            present = [
                v for v in vals
                if v is not None
            ]
            if not present:
                return vals
            p0 = present[0]
            if isinstance(p0, str):
                return [
                    "unknown" if v is None else v
                    for v in vals
                ]
            if isinstance(p0, (int, np.integer)):
                return torch.as_tensor(
                    [
                        -1 if v is None else int(v)
                        for v in vals
                    ],
                    dtype=torch.long,
                )
            if isinstance(p0, (float, np.floating)):
                return torch.as_tensor(
                    [
                        float("nan") if v is None else float(v)
                        for v in vals
                    ],
                    dtype=torch.float32,
                )
            return vals

        if modality == "wav":
            return vals

        v0 = vals[0]

        # ---------------------------------------------
        # Structured dicts
        # ---------------------------------------------
        if isinstance(v0, dict):
            keys = tuple(v0.keys())

            for value in vals[1:]:
                if tuple(value.keys()) != keys:
                    raise ValueError(
                        f"Inconsistent dictionary structure for modality "
                        f"'{modality}'"
                    )

            return {
                key: self._assemble(
                    [value[key] for value in vals],
                    modality=f"{modality}.{key}",
                )
                for key in keys
            }
        # ---------------------------------------------
        # numpy arrays
        # ---------------------------------------------

        if isinstance(v0, np.ndarray):

            return torch.from_numpy(
                np.stack(vals)
            ).to(torch.float32)

        # ---------------------------------------------
        # strings
        # ---------------------------------------------

        if isinstance(v0, str):
            return vals

        # ---------------------------------------------
        # scalar labels / ids
        # ---------------------------------------------

        if isinstance(v0, (int, np.integer)):
            return torch.as_tensor(
                vals,
                dtype=torch.long,
            )

        if isinstance(v0, (float, np.floating)):
            return torch.as_tensor(
                vals,
                dtype=torch.float32,
            )

        return vals
