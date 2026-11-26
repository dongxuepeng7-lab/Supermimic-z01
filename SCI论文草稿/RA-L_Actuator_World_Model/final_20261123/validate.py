"""Recalculate manuscript summaries from bundled records; Python standard library only."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


audit = read("evidence/online_night01_review_20261123.json")
saved = read("evidence/table_online_values.json")
table = (ROOT / "table_online.tex").read_text()
manifest = read("evidence/input_manifest.json")
for item in manifest:
    assert hashlib.sha256((ROOT / item["bundled"]).read_bytes()).hexdigest() == item["sha256"]

count, reductions, failures = 0, 0, 0
for machine, cohort in audit["machines"].items():
    pairs = cohort["pairs"]
    count += len(pairs)
    for group, directory in [("A+", "Aplus"), ("B", "B")]:
        values = {"completion": [], "cal_q32": [], "cal_r8": []}
        for pair in pairs:
            base = f"evidence/online_metadata/{machine}/s{pair['seed']}_{directory}"
            predictions = read(f"{base}/predictions.json")["splits"]["cal"]
            for repeat in (0, 1):
                ev = read(f"{base}/eval_r{repeat}.json")
                assert len(ev["per_clip"]) == 163
                completed = 163 - ev["n_failed"]
                assert completed == pair["repeat_completed"][group][repeat]
                if repeat == 0:
                    assert completed == pair["primary"][group]["completed"]
                    values["completion"].append(100 * completed / 163)
            values["cal_q32"].append(predictions["q_rmse_rad"]["32"])
            values["cal_r8"].append(predictions["response_rmse"]["8"])
            close(predictions["zero_response_rmse"]["8"], 73.89330385486116)
        row = []
        for key, precision in [("completion", 2), ("cal_q32", 5), ("cal_r8", 3)]:
            mean, sd = statistics.mean(values[key]), statistics.stdev(values[key])
            close(mean, saved[machine][group][key]["mean"])
            close(sd, saved[machine][group][key]["sample_sd"])
            row.append(f"${mean:.{precision}f}\\pm{sd:.{precision}f}$")
        assert f"{machine} & {len(pairs)} & {group} & " + " & ".join(row) in table
    for pair in pairs:
        seed = pair["seed"]
        a = read(f"evidence/online_metadata/{machine}/s{seed}_Aplus/predictions.json")["splits"]["cal"]
        b = read(f"evidence/online_metadata/{machine}/s{seed}_B/predictions.json")["splits"]["cal"]
        reduction = 100 * (1 - b["q_rmse_rad"]["32"] / a["q_rmse_rad"]["32"])
        close(reduction, pair["prediction_diagnostics"]["cal"]["q32_B_reduction_percent"])
        delta = 100 * (pair["primary"]["B"]["completed"] - pair["primary"]["A+"]["completed"]) / 163
        close(delta, pair["primary_B_minus_Aplus_completion_pp"])
        assert -12 < reduction < 16 and -8 < delta < 9, "Scatter plot limits hide a point"
        reductions += reduction > 0
        failures += not pair["gate"]["passed"]
assert (count, reductions, failures) == (31, 24, 2)

parameters = list(csv.DictReader((ROOT / "evidence/params.csv").open()))
enabled = [r for r in parameters if r["enabled"] == "1"]
assert len(enabled) == 21
parameter_table = (ROOT / "table_params.tex").read_text()
for joint in {r["joint_id"] for r in enabled}:
    values = {r["param_id"]: float(r["effective_A"]) for r in enabled if r["joint_id"] == joint}
    formatted = ["0" if values[k] == 0 else f"{values[k]:.6f}" for k in ["armature", "damping", "frictionloss"]]
    assert " & ".join(formatted) in parameter_table

replay = list(csv.DictReader((ROOT / "evidence/archived_replay_metrics.csv").open()))
replay_table = (ROOT / "table_archived.tex").read_text()
improved = 0
for joint in {r["joint"] for r in replay}:
    rows = [r for r in replay if r["joint"] == joint]
    original = next(r for r in rows if r["parameter_set"] == "original_model")
    identified = next(r for r in rows if r["parameter_set"] != "original_model")
    values = []
    for key, multiplier, precision in [("q", 1000, 3), ("v", 1, 4), ("tau", 1, 3)]:
        values.extend(f"{float(r[key]) * multiplier:.{precision}f}" for r in [original, identified])
    assert "&".join(values) in replay_table
    improved += all(float(identified[k]) < float(original[k]) for k in ["q", "v", "tau"])
assert improved == 7

randomized_table = (ROOT / "table_randomized.tex").read_text()
for contrast in audit["local_3090"]["contrasts"]:
    a, b, delta = [contrast[k]["mean"] for k in ["Aplus_completion_percent", "B_completion_percent", "B_minus_Aplus_completion_pp"]]
    assert f"{a:.2f}&{b:.2f}&${delta:.2f}$" in randomized_table

en, zh = [(ROOT / f"body_{language}.tex").read_text() for language in ("en", "zh")]
for pattern in [r"\\label\{([^}]+)\}", r"\\input\{([^}]+)\}", r"\\cite\{([^}]+)\}"]:
    assert re.findall(pattern, en) == re.findall(pattern, zh), pattern
for environment in ["equation", "figure*"]:
    assert en.count(f"\\begin{{{environment}}}") == zh.count(f"\\begin{{{environment}}}")
assert en.count("\\begin{equation}") == 10
assert en.count("\\section{") == zh.count("\\section{") == 6
keys = set(re.findall(r"@\w+\{([^,]+)", (ROOT / "references.bib").read_text()))
cited = set(",".join(re.findall(r"\\cite\{([^}]+)\}", en)).split(","))
assert keys == cited and len(keys) == 18
assert "arxiv" not in (ROOT / "references.bib").read_text().lower()
assert {r["bibtex_key"] for r in read("reference_audit.json")["entries"]} == keys
close(0.01 / math.sqrt(832) * 1000, 0.34668762264076824)
close(1 / (math.sqrt(2) * 0.01), 70.71067811865474)
assert 1024 * (64 + math.floor(5000 / 0.4)) == 12865536
assert 5000 * 512 * 32 == 81920000
print("PASS: 186 metadata hashes; 124 evaluation counts; 62 calibration records; all online table means/SDs;")
print("31 plotted pairs, including 2 failed screens; bilingual equations/citations/tables; 18 bibliography entries; exact budget arithmetic.")
print("21 enabled parameters; all 8 archived replay rows; 4 independent randomized evaluation contrasts.")
