#!/usr/bin/env python3
"""Summarize/plot paired-noise probes without loading models or using a GPU.

Accepts existing precision_probe summary.json or metrics.jsonl, and extended
rows with sublayer stages. Each forward row should contain branch,
requested_time, metrics, and activations: {stage_name: {shape, dtype,
action_tokens: {rms, delta_rms, delta_relative_rms, ...}, all_tokens: {...}}}.
Optional stage metadata: block_index, attention_type ('cross'/'self'), component.
Recommended sublayer stage names: block_00.input, block_00.norm1,
block_00.attn_output, block_00.after_attn, block_00.ff_output, block_00.output.
Optional row diagnostics (e.g. attention entropy) are retained in summary.json.

Example:
  python3 analyze_layer_probe.py --input ../xperience_repair_validation/precision_ln600/summary.json --output report
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/xperience-layer-probe-mpl")


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def divide(a, b):
    return a / b if a is not None and b is not None and abs(b) > 1e-30 else None


def read_rows(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    document = json.loads(path.read_text())
    if isinstance(document, list):
        return document
    return document.get("results", document.get("rows", [document]))


def stage_metadata(name, item):
    nested = item.get("blockmetadata", item.get("block_metadata", item.get("metadata", {})))
    item = {**nested, **item}
    match = re.match(r"^block[_-]?(\d+)(?:[._/-](.*))?$", name)
    index = item.get("block_index", item.get("index", int(match.group(1)) if match else None))
    component = item.get("component", (match.group(2) or "output") if match else name)
    attention = item.get("attention_type", ("cross" if index % 2 == 0 else "self") if index is not None else "none")
    return index, component, attention


def flatten(rows, path, scope):
    groups = []
    for row in rows:
        if "activations" not in row:
            continue
        condition = str(row.get("variant", row.get("condition", row.get("checkpoint", path.parent.name))))
        branch = str(row.get("branch", row.get("precision", "unspecified")))
        t = number(row.get("requested_time", row.get("time", row.get("t"))))
        stages = []
        for name, raw in row["activations"].items():
            stats = raw.get(scope, raw.get("stats", raw))
            rms = number(stats.get("rms"))
            delta = number(stats.get("delta_rms"))
            if rms is None or delta is None:
                continue
            index, component, attention = stage_metadata(name, raw)
            relative = number(stats.get("delta_relative_rms"))
            if relative is None:
                relative = divide(delta, rms)
            stages.append({"condition": condition, "branch": branch, "time": t,
                "source": str(path.resolve()), "stage": name, "block_index": index,
                "round": row.get("round"), "seed": row.get("seed"),
                "component": component, "attention_type": attention,
                "rms": rms, "delta_rms": delta, "delta_relative_rms": relative,
                # Exact identity for two equally weighted paired populations.
                "paired_common_mode_rms": math.sqrt(max(0.0, rms * rms - delta * delta / 4)),
                "paired_exactly_equal_fraction": number(stats.get("paired_exactly_equal_fraction")),
                "centered_pair_cosine": number(stats.get("centered_pair_cosine")),
                "common_rms": number(stats.get("common_rms")),
                "positive_fraction": number(stats.get("positive_fraction")),
                "finite": stats.get("finite"), "dtype": raw.get("dtype"),
                "shape": raw.get("shape"), "max_abs": number(stats.get("max_abs"))})
        if stages:
            groups.append({"condition": condition, "branch": branch, "time": t,
                "source": str(path.resolve()), "stages": stages,
                "round": row.get("round"), "seed": row.get("seed"),
                "num_observations": row.get("num_observations"), "quantized_time": row.get("quantized_time"),
                "metrics": row.get("metrics", {}), "diagnostics": row.get("diagnostics", {})})
    return groups


def block_outputs(group):
    return sorted([s for s in group["stages"] if s["block_index"] is not None
                   and s["component"] in ("output", "block_output", "out")], key=lambda s: s["block_index"])


def summarize(group, late_start):
    stages = {item["stage"]: item for item in group["stages"]}
    blocks = block_outputs(group)
    result = {key: group.get(key) for key in ("condition", "branch", "time", "source", "round", "seed",
        "num_observations", "quantized_time", "metrics", "diagnostics")}
    result["block_output_count"] = len(blocks)
    if not blocks:
        result["warning"] = "No block output stages found; sublayer stages retained in stages.csv"
        return result
    peak = max(blocks, key=lambda s: s["delta_relative_rms"] or -math.inf)
    last = blocks[-1]
    result.update(final_block=last, peak_relative_delta_block=peak["block_index"],
        peak_relative_delta=peak["delta_relative_rms"],
        final_relative_delta_over_peak=divide(last["delta_relative_rms"], peak["delta_relative_rms"]))
    previous = stages.get("dit_input", stages.get("block_00.input"))
    growth = []
    for item in blocks:
        if previous is not None:
            growth.append({"block_index": item["block_index"], "attention_type": item["attention_type"],
                "scalar_rms_change": item["rms"] - previous["rms"],
                "scalar_delta_rms_change": item["delta_rms"] - previous["delta_rms"],
                "rms_gain": divide(item["rms"], previous["rms"]),
                "delta_gain": divide(item["delta_rms"], previous["delta_rms"]),
                "relative_delta_gain": divide(item["delta_relative_rms"], previous["delta_relative_rms"])})
        previous = item
    result["per_block_scalar_changes"] = growth
    late = [row for row in growth if row["block_index"] >= late_start]
    totals = {kind: sum(row["scalar_rms_change"] for row in late if row["attention_type"] == kind)
              for kind in ("cross", "self")}
    result["late_scalar_rms_changes"] = {"start_block_zero_based": late_start, **totals,
        "cross_fraction": divide(totals["cross"], totals["cross"] + totals["self"]),
        "interpretation": "Coincidence by block type only. Changes in RMS are not additive sublayer energies or proof that attention rather than FF caused growth."}
    reference = next((item for item in blocks if item["block_index"] == late_start - 1), None)
    if reference is not None:
        result["late_to_final"] = {"rms_gain": divide(last["rms"], reference["rms"]),
            "delta_rms_gain": divide(last["delta_rms"], reference["delta_rms"]),
            "relative_delta_gain": divide(last["delta_relative_rms"], reference["delta_relative_rms"])}
    result["decoder_stages"] = {k: v for k, v in stages.items() if k.startswith("decoder")}
    decoder_order = ["decoder_input", "decoder_after_input_norm", "decoder_pre_relu",
                     "decoder_layer1_pre_relu", "decoder_post_relu", "decoder_output"]
    decoder = [stages[key] for key in decoder_order if key in stages]
    result["decoder_gains"] = [{"from": before["stage"], "to": after["stage"],
        "rms_gain": divide(after["rms"], before["rms"]),
        "delta_rms_gain": divide(after["delta_rms"], before["delta_rms"]),
        "relative_delta_gain": divide(after["delta_relative_rms"], before["delta_relative_rms"])}
        for before, after in zip(decoder, decoder[1:])]
    # Optional new instrumentation: measure update amplitude relative to residual.
    result["sublayer_update_ratios"] = []
    for index in sorted({s["block_index"] for s in group["stages"] if s["block_index"] is not None}):
        parts = {s["component"]: s for s in group["stages"] if s["block_index"] == index}
        for numerator, denominator in (("attn_output", "input"), ("ff_output", "after_attn")):
            if numerator in parts and denominator in parts:
                update, residual = parts[numerator], parts[denominator]
                result["sublayer_update_ratios"].append({"block_index": index, "update": numerator,
                    "residual": denominator, "rms_ratio": divide(update["rms"], residual["rms"]),
                    "delta_rms_ratio": divide(update["delta_rms"], residual["delta_rms"])})
    return result


def precision_comparisons(groups):
    paired = {}
    for group in groups:
        paired.setdefault((group["condition"], group["time"]), {})[group["branch"]] = group
    result = []
    for (condition, time), branches in paired.items():
        if not {"bf16", "fp32"}.issubset(branches):
            continue
        a, b = branches["bf16"], branches["fp32"]
        shared = sorted(set(a["metrics"]) & set(b["metrics"]))
        result.append({"condition": condition, "time": time,
            "fp32_minus_bf16": {key: float(b["metrics"][key]) - float(a["metrics"][key]) for key in shared
                if isinstance(a["metrics"][key], (int, float)) and not isinstance(a["metrics"][key], bool)},
            "bf16": a["metrics"], "fp32": b["metrics"]})
    return result


def time_aggregates(groups):
    """Across-forward summaries; never mistake these for within-pair variance."""
    grouped = {}
    for group in groups:
        grouped.setdefault((group["condition"], group["branch"], group["time"]), []).append(group)
    out = []
    for (condition, branch, time), records in grouped.items():
        names = sorted(set().union(*(r["metrics"].keys() for r in records)))
        metrics = {}
        for name in names:
            values = [number(r["metrics"].get(name)) for r in records]
            values = [v for v in values if v is not None]
            if values:
                metrics[name] = aggregate_values(values)
        stage_rows = {}
        for record in records:
            for stage in record["stages"]:
                stage_rows.setdefault(stage["stage"], []).append(stage)
        stages = {}
        for stage_name, observations in stage_rows.items():
            fields = {}
            for name in ("rms", "delta_rms", "delta_relative_rms", "paired_common_mode_rms",
                         "common_rms", "positive_fraction", "centered_pair_cosine", "paired_exactly_equal_fraction"):
                values = [number(row.get(name)) for row in observations]
                values = [v for v in values if v is not None]
                if values:
                    fields[name] = aggregate_values(values)
            stages[stage_name] = fields
        diagnostic_rows = {}
        for record in records:
            for stage_name, observations in record["diagnostics"].items():
                if not isinstance(observations, dict):
                    continue
                for field, value in observations.items():
                    value = number(value)
                    if value is not None:
                        diagnostic_rows.setdefault(stage_name, {}).setdefault(field, []).append(value)
        diagnostics = {stage: {field: aggregate_values(values) for field, values in fields.items()}
                       for stage, fields in diagnostic_rows.items()}
        out.append({"condition": condition, "branch": branch, "time": time, "metrics": metrics,
                    "stages": stages, "diagnostics": diagnostics,
                    "note": "Pair/sample variances, if measured by the producer, remain in the original metrics/diagnostics. These aggregates are only across forward records."})
    return out


def aggregate_values(values):
    return {"mean_across_forward_records": statistics.mean(values),
            "population_variance_across_forward_records": statistics.pvariance(values),
            "sample_variance_across_forward_records": statistics.variance(values) if len(values) > 1 else None,
            "count": len(values)}


def aggregate_plot_groups(groups):
    """Plot one mean curve per condition/time, retaining individual rows in outputs."""
    aggregates = time_aggregates(groups)
    indexed = {(a["condition"], a["branch"], a["time"]): a for a in aggregates}
    result, seen = [], set()
    for group in groups:
        key = (group["condition"], group["branch"], group["time"])
        if key in seen:
            continue
        seen.add(key)
        aggregate = indexed[key]
        stages = []
        for stage in group["stages"]:
            means = {field: value["mean_across_forward_records"] for field, value in aggregate["stages"][stage["stage"]].items()}
            stages.append({**stage, **means})
        result.append({**group, "stages": stages,
            "metrics": {field: value["mean_across_forward_records"] for field, value in aggregate["metrics"].items()},
            "diagnostics": {stage: {field: value["mean_across_forward_records"] for field, value in fields.items()}
                            for stage, fields in aggregate["diagnostics"].items()}})
    return result


def make_plots(groups, summaries, output, scope):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("tab10")
    times = sorted({g["time"] for g in groups if g["time"] is not None})
    conditions = sorted({g["condition"] for g in groups})
    color_key = {(condition, time): colors(i % 10) for i, (condition, time) in enumerate(
        (c, t) for c in conditions for t in times)}
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for g, summary in zip(groups, summaries):
        blocks = block_outputs(g)
        if not blocks:
            continue
        x = [s["block_index"] for s in blocks]
        color = color_key.get((g["condition"], g["time"]), colors(0))
        linestyle = "--" if g["branch"] == "fp32" else "-"
        label = f"{g['condition']} {g['branch']} t={g['time']}"
        for ax, key in ((axes[0, 0], "rms"), (axes[0, 1], "delta_rms"), (axes[1, 0], "delta_relative_rms")):
            ax.plot(x, [max(s[key] or 0, 1e-12) for s in blocks], color=color, linestyle=linestyle, label=label)
        changes = summary.get("per_block_scalar_changes", [])
        axes[1, 1].plot([v["block_index"] for v in changes], [v["scalar_rms_change"] for v in changes],
                        color=color, linestyle=linestyle, label=label)
        by_stage = {s["stage"]: s for s in g["stages"]}
        by_index = {s["block_index"]: s for s in blocks}
        selected = [by_stage.get("dit_input", by_stage.get("block_00.input", {})), by_index.get(11, {}),
                    by_index.get(35, {}), by_stage.get("decoder_after_input_norm", {}),
                    by_stage.get("decoder_pre_relu", by_stage.get("decoder_layer1_pre_relu", {}))]
        axes[2, 0].plot(range(len(selected)), [s.get("delta_relative_rms", math.nan) for s in selected],
                       color=color, linestyle=linestyle, marker="o", label=label)
        metrics = g["metrics"]
        if "noise_delta_cosine" in metrics and "noise_delta_relative_mse" in metrics:
            axes[2, 1].scatter(metrics["noise_delta_cosine"], metrics["noise_delta_relative_mse"], color=color,
                              marker="x" if g["branch"] == "fp32" else "o", label=label)
            axes[2, 1].annotate(f"t={g['time']} {g['branch']}", (metrics["noise_delta_cosine"], metrics["noise_delta_relative_mse"]),
                              xytext=(4, 4 if g["branch"] == "bf16" else -10), textcoords="offset points", fontsize=7)
    for ax in axes[:2].flat:
        for index in range(0, 36, 2):
            ax.axvspan(index - .45, index + .45, color="#6485aa", alpha=.06, zorder=-2)
        ax.set_xlabel("DiT block index (zero based; shaded = cross type)")
        ax.grid(alpha=.18)
    axes[0, 0].set(title="Activation RMS", ylabel="RMS", yscale="log")
    axes[0, 1].set(title="Paired-noise difference magnitude", ylabel="RMS of h(noise B) - h(noise A)", yscale="log")
    axes[1, 0].set(title="Relative paired difference", ylabel="delta RMS / activation RMS", yscale="log")
    axes[1, 1].set(title="Change in scalar RMS across each whole block", ylabel="RMS(output) - RMS(input)")
    axes[2, 0].set(title="Selected stages: relative paired difference", ylabel="delta RMS / activation RMS")
    axes[2, 0].set_xticks(range(5), ["DiT input", "block 11", "block 35", "decoder LN", "linear1 pre-ReLU"], rotation=18)
    axes[2, 1].set(title="Final output: alignment and error", xlabel="Noise-delta cosine (higher is better)",
                   ylabel="Noise-delta relative MSE (lower is better)")
    axes[2, 1].margins(x=.22, y=.15)
    axes[0, 0].legend(fontsize=7, loc="best")
    for ax in axes[2]:
        ax.grid(alpha=.18)
    fig.suptitle(f"Paired-noise propagation | {scope}\nMeans across forward records; same observation, action and time within each noise pair", fontsize=14)
    fig.savefig(output / "layer_propagation.png", dpi=160)
    fig.savefig(output / "layer_propagation.svg")
    plt.close(fig)
    # Extended probe data: plot actual sublayer update/residual ratios if supplied.
    extended = [s for s in summaries if s.get("sublayer_update_ratios")]
    if extended:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
        for summary in extended:
            for update in ("attn_output", "ff_output"):
                rows = [r for r in summary["sublayer_update_ratios"] if r["update"] == update]
                label = f"{summary['condition']} {summary['branch']} t={summary['time']} {update}"
                for ax, key in zip(axes, ("rms_ratio", "delta_rms_ratio")):
                    ax.plot([r["block_index"] for r in rows], [r[key] for r in rows], label=label)
                    ax.set(xlabel="Block index", ylabel=key, yscale="log")
                    ax.grid(alpha=.18)
        axes[0].legend(fontsize=6)
        fig.savefig(output / "sublayer_update_ratios.png", dpi=160)
        fig.savefig(output / "sublayer_update_ratios.svg")
        plt.close(fig)
    detailed = [g for g in groups if g["time"] is not None and abs(g["time"] - .5) < 1e-8
                and any(s["component"] == "attn_output" for s in g["stages"])]
    if detailed:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        decoder_order = ["decoder_input", "decoder_after_input_norm", "decoder_pre_relu",
                         "decoder_layer1_pre_relu", "decoder_post_relu", "decoder_output"]
        for group_index, g in enumerate(detailed):
            label = f"{g['condition']} {g['branch']}"
            color = colors(group_index % 10)
            for component, style in (("attn_output", "-"), ("ff_output", "--")):
                rows = sorted([s for s in g["stages"] if s["component"] == component], key=lambda s: s["block_index"])
                for ax, field in zip(axes[0, :2], ("rms", "delta_rms")):
                    ax.plot([r["block_index"] for r in rows], [r[field] for r in rows],
                            color=color, linestyle=style, label=label + " " + component)
                    for kind, marker in (("cross", "o"), ("self", "x")):
                        part = [r for r in rows if r["attention_type"] == kind]
                        ax.scatter([r["block_index"] for r in part], [r[field] for r in part],
                                   color=color, marker=marker, s=18)
            summary = summarize(g, 12)
            for component, style in (("attn_output", "-"), ("ff_output", "--")):
                ratios = [r for r in summary["sublayer_update_ratios"] if r["update"] == component]
                axes[0, 2].plot([r["block_index"] for r in ratios], [r["rms_ratio"] for r in ratios],
                               color=color, linestyle=style, label=label + " " + component)
            blocks = block_outputs(g)
            axes[1, 0].plot([r["block_index"] for r in blocks], [r["delta_relative_rms"] for r in blocks],
                           color=color, label=label)
            by_stage = {s["stage"]: s for s in g["stages"]}
            decoder = [by_stage[name] for name in decoder_order if name in by_stage]
            for key, style in (("rms", "-"), ("delta_rms", "--")):
                axes[1, 1].plot(range(len(decoder)), [s[key] for s in decoder], color=color,
                               linestyle=style, marker="o", label=label + " " + key)
            axes[1, 2].plot(range(len(decoder)), [s["delta_relative_rms"] for s in decoder],
                           color=color, marker="o", label=label)
            labels = [s["stage"].replace("decoder_", "").replace("layer1_", "") for s in decoder]
            for ax in axes[1, 1:]:
                ax.set_xticks(range(len(decoder)), labels, rotation=25)
        titles = ["Sublayer update RMS", "Sublayer paired-difference RMS", "Update / residual RMS",
                  "Block output: delta RMS / RMS", "Decoder absolute magnitudes", "Decoder relative paired difference"]
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)
            ax.set_yscale("log")
            ax.grid(alpha=.18)
        for ax in list(axes[0]) + [axes[1, 0]]:
            ax.set_xlabel("Block index; circle=cross type, x=self type")
        axes[0, 0].legend(fontsize=7)
        axes[1, 1].legend(fontsize=7)
        fig.suptitle(f"t=0.5 sublayer / decoder decomposition | {scope}\nAttention and FF measured before their residual additions", fontsize=14)
        fig.savefig(output / "decomposition_t0p5.png", dpi=160)
        fig.savefig(output / "decomposition_t0p5.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("action_tokens", "prefix_tokens", "all_tokens"), default="action_tokens")
    parser.add_argument("--time", type=float, help="Optional filter for a single time, e.g. 0.5")
    parser.add_argument("--late-start", type=int, default=12)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_rows = [(path, read_rows(path)) for path in args.input]
    groups = [group for path, rows in source_rows for group in flatten(rows, path, args.scope)]
    if args.time is not None:
        groups = [g for g in groups if g["time"] is not None and abs(g["time"] - args.time) < 1e-8]
    if not groups:
        raise SystemExit("No valid activation rows; see the documented JSONL schema")
    summaries = [summarize(group, args.late_start) for group in groups]
    cautions = [
        "A large or small paired hidden difference measures sensitivity, not task-relevant information or recoverability.",
        "Common-mode RMS is a paired algebraic decomposition; it is not a pure VLM-conditioning or bias measurement.",
        "Cross/self type inferred from even/odd block index only when attention_type is absent; this matches the inspected interleaved QwenPI configuration.",
        "Block outputs do not identify attention-versus-FF contributions, softmax saturation, or residual cancellation; sublayer/attention diagnostics are required.",
        "An inference precision comparison cannot establish that training in another precision would have learned the same solution."]
    document = {"scope": args.scope, "inputs": [str(p.resolve()) for p in args.input],
        "groups": summaries, "precision_comparisons": precision_comparisons(aggregate_plot_groups(groups)),
        "by_time": time_aggregates(groups), "interpretation_limits": cautions,
        "duplicate_controls": [row for _, rows in source_rows for row in rows if row.get("kind") == "duplicate_control"]}
    (args.output / "summary.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    stages = [stage for group in groups for stage in group["stages"]]
    with (args.output / "stages.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stages[0]))
        writer.writeheader()
        writer.writerows(stages)
    if not args.no_plot:
        plot_groups = aggregate_plot_groups(groups)
        make_plots(plot_groups, [summarize(group, args.late_start) for group in plot_groups], args.output, args.scope)
    print(json.dumps({"groups": len(groups), "stages": len(stages), "output": str(args.output.resolve()),
                      "plots": not args.no_plot}, indent=2))


if __name__ == "__main__":
    main()
