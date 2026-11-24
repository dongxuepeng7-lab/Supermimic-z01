"""Read-only evidence audit of diag01; never loads model tensors or locked test arrays.

Imported remote metadata and local 3090 raw trajectories are audited separately.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

from ral_eight_hour import ROOT, atomic_json, digest, read
from ral_nine_hour import CONFIG, PROTOCOL, factorial_result, verify_profile_inputs
from ral_online_night import compare_backend_processes
from ral_recovery import check_run, evaluation_summary
from ral_structure import stage_gate
from ral_backend_diagnose import compare, rms


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b, location=""):
    if isinstance(a, dict) and isinstance(b, dict):
        require(a.keys() == b.keys(), f"Keys differ: {location}")
        for key in a:
            equal(a[key], b[key], location+"/"+str(key))
    elif isinstance(a, list) and isinstance(b, list):
        require(len(a) == len(b), f"Lengths differ: {location}")
        for i, (x,y) in enumerate(zip(a,b)):
            equal(x,y,location+"/"+str(i))
    elif isinstance(a, (int,float)) and not isinstance(a,bool) and isinstance(b, (int,float)) and not isinstance(b,bool):
        require(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),f"Value differs: {location}: {a} != {b}")
    else:
        require(a == b,f"Value differs: {location}")


def stats(values):
    require(bool(values), "Empty summary")
    return dict(n=len(values), mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values), maximum=max(values))


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite JSON value")
    elif isinstance(value, dict):
        for item in value.values(): finite_tree(item)
    elif isinstance(value, list):
        for item in value: finite_tree(item)


def audit_machine(root, plan, protocol):
    state, judgment = read(root/"status.json"), read(root/"judgment.json")
    require(read(root/"protocol.json")==protocol, "Remote frozen protocol differs")
    require(read(root/"config.json")==plan, "Remote configuration differs")
    require(read(root/"inputs.json")["tools"]==protocol["tools"], "Remote tool hashes differ")
    report = dict(machine=state["machine"], status=state["status"], source=str(root.relative_to(ROOT)),
        wall_hours=(state["finished_unix"]-state["started_unix"])/3600, disk=state["disk_check"],
        complete_models=0, checked_iteration_rows=0, checked_metadata_hashes=0, checked_evaluations=0,
        checked_episodes=0, actual_model_weights_present=False, models=[], prior=[], complete_blocks=[], incomplete_blocks=[])
    d=state["disk_check"]
    report["storage_stop_explanation"] = dict(free_GiB=d["free_bytes"]/2**30, output_GiB=d["output_bytes"]/2**30,
        free_below_17_GiB_start_reserve=d["free_bytes"] < 17*2**30,
        output_plus_2_GiB_reaches_16_GiB_cap=d["output_bytes"]+2*2**30 >= 16*2**30)

    def evaluation(folder, name, checkpoint_hash):
        result=read(folder/(name+".json")); metadata=read(folder/(name+".inputs.json"))
        require(result["plant"]=="identified" and metadata["plant"]=="identified", "Unexpected evaluation plant")
        require(metadata["eval_seed"]==0 and metadata["checkpoint_weights"]=="EMA", "Unexpected evaluation setup")
        require(metadata["checkpoint_sha256"]==checkpoint_hash, "Evaluation checkpoint differs")
        for key in ("code_sha256","asset_sha256","params_sha256","response_sha256","motion_sha256"):
            require(metadata["inputs"][key]==protocol[key], "Evaluation inputs differ: "+key)
        summary=evaluation_summary(result,protocol["motions"])
        report["checked_evaluations"]+=1;report["checked_episodes"]+=len(result["episodes"])
        return summary

    def gradient(folder, record, checkpoint_hash):
        path=folder/"gradient.json"
        require(digest(path)==record["sha256"], "Gradient hash differs")
        data=read(path);finite_tree(data)
        require(data["checkpoint_sha256"]==checkpoint_hash and data["status"]=="completed" and data["all_finite"], "Gradient provenance invalid")
        require(not data["response_test_accessed"] and len(data["blocks"])==48, "Gradient coverage differs")
        for split in ("fit","cal"):
            rows=[b for b in data["blocks"] if b["split"]==split]
            require(len(rows)==24 and len({r["block_id"] for r in rows})==24, "Duplicate/missing gradient blocks")
            require(all(r["valid"] and set(r["horizons"])=={"1","2","4","8","16","32"} for r in rows), "Gradient horizons differ")
        cal=[b for b in data["blocks"] if b["split"]=="cal"]
        return {h:{key:statistics.median(b["horizons"][h][key] for b in cal)
            for key in cal[0]["horizons"][h] if all(b["horizons"][h][key] is not None for b in cal)} for h in ("1","8","32")}

    for block in state["blocks"]:
        hashes=[]
        for condition in block["conditions"]:
            if not condition.get("folder"): continue
            folder=root/condition["folder"]
            cp=read(folder/"protocol.json");cfg=read(folder/"config.json")
            expected=dict(plan["training"], **{block["factor"]:condition["level"]})
            require(cp["training"]==cfg["training"]==expected, "Unexpected training intervention")
            require(cp["config_sha256"]==digest(folder/"config.json"),"Condition config checksum differs")
            retention=condition["retention"]
            require(retention==read(folder/(condition["id"]+"_retention.json")), "Retention records differ")
            for name,sha in retention["members"].items():
                if Path(name).suffix==".pt":continue
                require(digest(folder/name)==sha,"Retained metadata changed: "+name)
                report["checked_metadata_hashes"]+=1
            for job in condition["jobs"]:
                if job.get("status")!="completed":continue
                where=folder/"jobs"/job["id"];run=where/"run"
                ledger,index=check_run(run,cp)
                require(ledger["protocol_sha256"]==digest(folder/"protocol.json"),"Run protocol differs")
                require((ledger["training_seed"],ledger["group"],ledger["policy_updates"],ledger["wm_updates"],ledger["env_steps"])
                    ==(block["seed"],job["group"],5000,25000,12865536),"Wrong training budget")
                require(ledger["initial_module_sha256"]==job["initial_module_sha256"],"Initial module metadata differs")
                hashes.append({k:ledger["initial_module_sha256"][k] for k in ("pi","critic")})
                with (run/"train_metrics.csv").open() as f:rows=list(csv.DictReader(f))
                require([int(r["iteration"]) for r in rows]==list(range(5001)),"Iteration coverage differs")
                for row in rows[1:]:
                    i=int(row["iteration"])
                    require(int(row["policy_updates"])==i and int(row["wm_updates"])==5*i,"Update counter differs")
                    require(int(row["imagined_steps"])==i*expected["pi_batch"]*expected["horizon"],"Imagined budget differs")
                    require(all(math.isfinite(float(row[k])) for k in ("wm_loss","policy_loss")),"Nonfinite training loss")
                report["checked_iteration_rows"]+=len(rows)
                with (run/"train_log.csv").open() as f:train=list(csv.DictReader(f))
                require(all(float(r["nonfinite"])==0 and all(math.isfinite(float(r[k])) for k in ("wm_loss","pi_loss","v_loss","pi_gradnorm")) for r in train),"Training nonfinite/skipped steps")
                checkpoint_hash=index["ckpt_005000.pt"]["sha256"]
                require(job["final_sha256"]==checkpoint_hash,"Final checkpoint metadata differs")
                primary=None
                for repeat in (0,1):
                    value=evaluation(where,f"eval_r{repeat}",checkpoint_hash)
                    stored=next(r for r in condition["rows"] if r["group"]==job["group"] and r["repeat_check"]==repeat)
                    equal(value,{k:stored[k] for k in value})
                    if repeat==0:primary=value
                pred=read(where/"predictions.json");finite_tree(pred)
                require(pred==job["predictions"] and pred["checkpoint_sha256"]==checkpoint_hash and not pred["response_test_accessed"],"Prediction provenance differs")
                require(set(pred["splits"])=={"fit","cal"} and all(pred["splits"][s]["valid"] and pred["splits"][s]["blocks"]==n for s,n in (("fit",400),("cal",116))),"Prediction coverage differs")
                grad=gradient(where,job["gradient"],checkpoint_hash)
                term=read(where/"eval_r0.termination.json")
                require(dict(Counter(e["cause"] for e in term["episodes"]))=={k:v for k,v in term["counts"].items() if v},"Termination counts differ")
                require(term["counts"]["reference_complete"]==primary["completed"],"Termination completion differs")
                report["models"].append(dict(seed=block["seed"],level=condition["level"],group=job["group"],
                    included_in_balanced_comparison=block["status"]=="completed",primary=primary,
                    cal_predictions=pred["splits"]["cal"],cal_gradient_block_medians=grad,
                    termination_counts=term["counts"],final_checkpoint_sha256=checkpoint_hash,
                    training_seconds=ledger["finished_unix"]-ledger["started_unix"],
                    last_1000_update_mean={k:statistics.mean(float(r[k]) for r in train if int(r["iteration"])>4000) for k in ("wm_loss","pi_gradnorm","sim_rew","ep_len","term_frac")}))
                report["complete_models"]+=1
            if condition.get("status")=="completed":equal(stage_gate(condition["rows"],plan,seeds=[block["seed"]]),condition["gate"])
        require(all(x==hashes[0] for x in hashes),"Recorded four-model initialization differs")
        result=factorial_result(block,plan,state["machine"])
        if result is not None:report["complete_blocks"].append(result)
        else:report["incomplete_blocks"].append(dict(seed=block["seed"],complete_conditions=[c["id"] for c in block["conditions"] if c.get("status")=="completed"],excluded_from_balanced_comparison=True))
    equal(json.loads(json.dumps(report["complete_blocks"])),judgment["results"])
    for prior in state["prior"]:
        require(prior["status"]=="completed" and prior["actual_weights_verified"],"Prior diagnosis incomplete")
        rows=[]
        for job in prior["jobs"]:
            where=root/"prior"/f"s{prior['seed']}"/"jobs"/f"s{prior['seed']}_{job['group'].replace('+','plus')}"
            for repeat in (0,1):
                value=evaluation(where,f"reeval_r{repeat}",job["checkpoint_sha256"])
                equal(value,{k:job["rows"][repeat][k] for k in value})
            rows.append(dict(group=job["group"],rows=job["rows"],cal_gradient_block_medians=gradient(where,job["gradient"],job["checkpoint_sha256"])))
        report["prior"].append(dict(seed=prior["seed"],remote_reports_actual_weight_verification=True,locally_verified_tensor_bytes=False,jobs=rows))
    summary=[]
    for level in plan["machines"][state["machine"]]["levels"]:
        for group in ("A+","B"):
            rows=[r for r in report["models"] if r["level"]==level and r["group"]==group and r["included_in_balanced_comparison"]]
            summary.append(dict(level=level,group=group,completion_percent=stats([100*r["primary"]["completion"] for r in rows]),
                cal_q8_rmse_rad=stats([r["cal_predictions"]["q_rmse_rad"]["8"] for r in rows]),
                cal_q32_rmse_rad=stats([r["cal_predictions"]["q_rmse_rad"]["32"] for r in rows]),
                cal_response8_rmse=stats([r["cal_predictions"]["response_rmse"]["8"] for r in rows]),
                cal_gradient_mean_of_block_medians={h:{k:statistics.mean(r["cal_gradient_block_medians"][h][k] for r in rows) for k in rows[0]["cal_gradient_block_medians"][h]} for h in ("1","8","32")},
                termination_counts=dict(sum((Counter(r["termination_counts"]) for r in rows),Counter()))))
    report["summary"]=summary
    report["group_changes_pp"]={g:stats([b["group_changes_pp"][g] for b in report["complete_blocks"]]) for g in ("A+","B")}
    report["interaction_pp"]=stats([b["B_minus_Aplus_interaction_pp"] for b in report["complete_blocks"]])
    return report


def audit_backend(root):
    state=read(root/"status.json");require(read(root/"protocol.json")==read(PROTOCOL),"3090 protocol differs")
    result=dict(source=str(root.relative_to(ROOT)),status=state["status"],wall_hours=(state["finished_unix"]-state["started_unix"])/3600,
        full_original={},complete_direction_seeds=[],directions=[],incomplete_cells=[],raw_blocks_recomputed=0)
    for cell in state["backend"]:
        if cell["status"]!="completed":
            result["incomplete_cells"].append(dict(id=cell["id"],completed_jobs=sum(j.get("status")=="completed" for j in cell["jobs"]),excluded_from_balanced_comparison=True));continue
        folder=root/"backend"/cell["id"];profiles={}
        equal(verify_profile_inputs([folder/f"{p}_r0" for p in ("graph_batch","eager_batch","eager_single")]),cell["profile_inputs"])
        for profile in ("graph_batch","eager_batch","eager_single"):
            summaries=[]
            for repeat in (0,1):
                where=folder/f"{profile}_r{repeat}";record=next(j for j in cell["jobs"] if (j["profile"],j["repeat"])==(profile,repeat))
                require(digest(where/"summary.json")==record["summary_sha256"],"Backend summary hash differs")
                report=read(where/"summary.json");require(report["status"]=="completed" and not report["smoke_only"] and not report["response_test_accessed"],"Backend pass incomplete")
                paths={digest(p):p for p in where.glob("*_trajectories.npz")}
                categories=Counter();by_horizon=defaultdict(Counter);max_replay=max_transfer=max_capture=max_warm=0.
                for block in report["blocks"]:
                    with np.load(paths[block["trajectory_sha256"]]) as z:
                        for ck in ("train_cpu","train_gpu","pair_cpu","actions"):require(np.isfinite(z[ck]).all(),"Nonfinite raw trajectory")
                        comparisons=compare(z["train_cpu"],z["train_gpu"],z["scales"],z["epsilon"],report["config"])
                        archived=compare(z["pair_cpu"],z["train_gpu"],z["scales"],z["epsilon"],report["config"])
                        equal(comparisons,block["backend"]);equal(archived,block["archived_to_training"])
                        equal(rms((z["pair_cpu"]-z["train_cpu"])/z["scales"]),block["cpu_scene_scaled_rmse"])
                        equal(max(float(abs(z[n][0]-z[n][1]).max()) for n in ("pair_cpu","train_cpu")),block["cpu_repeat_max_abs"])
                        equal(bool(z["gpu_overflow"].any()),block["gpu_overflow_any"])
                        equal(int(z["gpu_solver_niter"].max()),block["gpu_solver_iterations_max"])
                        equal(int(z["gpu_nefc"].max()),block["gpu_constraints_max"])
                        categories["forward"]+=any(x["forward_scaled_rmse"]>report["config"]["forward_scaled_rmse_max"] for x in comparisons)
                        categories["CPU_GPU_response"]+=any(x["cross_response_rmse"]>x["response_limit"] for x in comparisons)
                        categories["within_process_repeat"]+=any(x["gpu_repeat_response_rmse"]>x["repeat_limit"] for x in comparisons)
                        for h in report["config"]["horizons"]:
                            a=[x for x in comparisons if x["horizon"]==h]
                            by_horizon[str(h)]["blocks"]+=1
                            by_horizon[str(h)]["failed_any_amplitude"]+=any(not x["passed"] for x in a)
                    max_replay=max(max_replay,block["saved_pair_max_abs"]);max_transfer=max(max_transfer,block["initial_state_transfer_max_abs"])
                    max_capture=max(max_capture,block["capture_state_mutation_max_abs"]);max_warm=max(max_warm,block["warmstart_transfer_max_abs"])
                    result["raw_blocks_recomputed"]+=1
                require(len(report["blocks"])==record["checked"] and sum(not x["passed"] for x in report["blocks"])==record["failed"],"Backend totals differ")
                summaries.append(dict(repeat=repeat,blocks=len(report["blocks"]),failed=record["failed"],failure_categories_overlapping=dict(categories),
                    by_horizon=dict(by_horizon),max_archived_replay_error_reported=max_replay,max_initial_transfer_error_reported=max_transfer,
                    max_capture_state_mutation_reported=max_capture,max_warmstart_transfer_error_reported=max_warm))
            cross=compare_backend_processes(folder/f"{profile}_r0",folder/f"{profile}_r1")
            equal(cross,read(folder/f"{profile}_cross_process.json"))
            summary=dict(checked=len(cross["blocks"]),failed=sum(not x["passed"] for x in cross["blocks"]))
            equal(summary,cell["cross_process"][profile])
            profiles[profile]=dict(passes=summaries,cross_process=summary)
            print(f"Verified 3090 {cell['id']} {profile}",flush=True)
        if cell["direction_seed"] is None:result["full_original"]=profiles
        else:
            result["complete_direction_seeds"].append(cell["direction_seed"])
            result["directions"].append(dict(seed=cell["direction_seed"],profiles=profiles))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"release/ral/diag01_review_20261124.json")
    a=p.parse_args()
    require(not a.out.exists(),"Use a new review output; never overwrite an audit")
    plan,protocol=read(CONFIG),read(PROTOCOL)
    result=dict(schema_version=1,date="2026-11-24",scope="Development diagnosis of diag01",formal_results=False,
        response_test_arrays_accessed=False,C_authorized=False,remote_model_tensor_bytes_verified_locally=False,
        audit_script_sha256=digest(__file__),frozen_protocol_sha256=digest(PROTOCOL),sources={},machines={})
    for machine in ("4090","5080"):
        for suffix in ("console.log","review.tar.gz"):
            path=Path("/home/zoyi/Downloads")/f"diag01_{machine}_{suffix}"
            result["sources"][str(path)]=digest(path)
        result["machines"][machine]=audit_machine(ROOT/"logs/ral_9h/import"/f"diag01_{machine}",plan,protocol)
        print(f"Verified {machine}: {result['machines'][machine]['complete_models']} model metadata records",flush=True)
    result["local_3090"]=audit_backend(ROOT/"logs/ral_9h/diag01_3090")
    result["limitations"]=["Remote review archives contain metadata, not tensor/model archive bytes. Actual weight checks are recorded remote assertions; local checks verify consistency and retained metadata hashes.",
        "Only complete four-model training blocks enter balanced comparison; completed conditions from a partial block remain separately recorded.",
        "First EMA evaluation is primary; second is repeatability, not an additional training seed.",
        "Different GPU/factor cohorts remain separate; these are training-reference development conditions, not heldout or hardware tests.",
        "Gradient summaries are means across models of 24-calibration-block medians, not whole-library response RMSE or full actor gradients.",
        "New-direction backend samples are failure-enriched and are not additional independent policy seeds or population failure estimates.",
        "CPU reconstruction metadata is retained; raw cross-backend/within-process/cross-process arithmetic was recomputed locally, not a newly executed simulator replay."]
    atomic_json(a.out,result)
    print(a.out)


if __name__=="__main__":
    main()
