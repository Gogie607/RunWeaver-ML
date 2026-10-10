"""Generate the README cosine illustration using RunWeaver schedule values.

Run from an environment with RunWeaver and Matplotlib available:
    python scripts/plot_readme_schedule.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runweaver_ml.phase_control import ScheduleEngine, TargetRegistry, build_schedule_block


def main():
    registry = TargetRegistry({"alignment_weight": 0.0})
    engine = ScheduleEngine(registry)
    engine.register("alignment_weight", build_schedule_block({
        "type": "cosine",
        "start": 0.0,
        "end": 1.0,
        "start_step": 0,
        "end_step": 1_000,
    }))

    steps = range(1_001)
    weights = []
    for step in steps:
        engine.update(step)
        weights.append(registry.get("alignment_weight"))

    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    ax.plot(steps, weights, color="#2563eb", linewidth=2.5)
    ax.set(title="A smooth start, a steady rise, a gentle finish",
           xlabel="Training step", ylabel="Alignment loss weight",
           xlim=(0, 1_000), ylim=(0, 1.05))
    ax.set_xticks([0, 250, 500, 750, 1_000])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    output = Path(__file__).resolve().parents[1] / "docs/images/cosine_schedule.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
