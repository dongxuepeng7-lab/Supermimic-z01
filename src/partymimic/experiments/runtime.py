"""Run provenance, immutable inputs and honest completion/interrupt bookkeeping."""

import atexit
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from .parameters import sha256


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def state_hash(modules):
    h = hashlib.sha256()
    for name, module in sorted(modules.items()):
        for key, value in sorted(module.state_dict().items()):
            value = value.detach().cpu().contiguous()
            h.update(f"{name}/{key}:{value.dtype}:{list(value.shape)}".encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def file_inventory(paths, root):
    return {str(Path(p).relative_to(root)): sha256(p) for p in sorted(paths)}


class RunLedger:
    def __init__(self, args, robot, modules):
        self.root = Path(args.log_dir)
        self.path = self.root / "run.json"
        if self.path.exists():
            raise ValueError(
                "Experiment run already exists; choose a new attempt directory"
            )
        repo = Path(__file__).resolve().parents[3]
        source = file_inventory(list((repo / "src").rglob("*.py")), repo)
        assets = file_inventory(
            [
                p
                for base in (repo / "assets/z01", repo / "Z01_v1.6_whole_body")
                for p in base.rglob("*")
                if p.is_file() and p.suffix.lower() != ".usd"
            ],
            repo,
        )
        motions = {
            p.name: sha256(p)
            for p in sorted(Path(args.motion_dir or robot.motion_dir).glob("*.npz"))
        }
        self.data = dict(
            schema_version=1,
            run_id=self.root.name,
            group=args.experiment_group,
            implementation="PartyMimic Z01 local fork (not unmodified SuperMimic)",
            training_seed=args.seed,
            status="running",
            args=vars(args),
            init_sha256=state_hash(modules),
            code_sha256=object_hash(source),
            asset_sha256=object_hash(assets),
            motion_sha256=object_hash(motions),
            params_sha256=sha256(args.parameter_manifest),
            response_sha256=sha256(args.response_manifest)
            if args.response_manifest
            else None,
            protocol_sha256=sha256(args.protocol) if args.protocol else None,
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            torch_version=torch.__version__,
            control_dt=0.02,
            physics_dt=0.005,
            step_definition="one environment transition at 50 Hz; includes collection warmup; excludes physics substeps",
            resume_semantics="new attempts only; original trainer resume is a warm restart, not exact continuation",
            wm_updates=0,
            policy_updates=0,
            env_steps=0,
            formal_results=False,
            started_unix=time.time(),
        )
        (self.root / "input_inventory.json").write_text(
            json.dumps(dict(code=source, assets=assets, motions=motions), indent=2)
        )
        from .parameters import nominal_parameters, read_manifest

        effective = nominal_parameters(robot)
        from partymimic.robot import get_robot

        rows = read_manifest(args.parameter_manifest, get_robot("z01"))
        with (self.root / "effective_params.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=(
                    "joint",
                    "parameter",
                    "unit",
                    "effective_value",
                    "enabled",
                    "source_ref",
                ),
            )
            w.writeheader()
            for r in rows:
                w.writerow(
                    dict(
                        joint=r["joint_id"],
                        parameter=r["param_id"],
                        unit=r["unit"],
                        effective_value=effective[r["joint_id"]][r["param_id"]],
                        enabled=r["enabled"],
                        source_ref=r["source_ref"],
                    )
                )
        (self.root / "config_diff.json").write_text(
            json.dumps(
                dict(
                    comparison="A relative to A0; all other plant parameters and controller gains unchanged",
                    enabled=[r for r in rows if r["enabled"] == "1"],
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        self.write()
        atexit.register(self.interrupted)
        self.metrics = (self.root / "train_metrics.csv").open("w", newline="")
        self.writer = csv.DictWriter(
            self.metrics,
            fieldnames=(
                "iteration",
                "env_steps",
                "wm_updates",
                "policy_updates",
                "imagined_steps",
                "elapsed_s",
                "wm_loss",
                "response_loss",
                "policy_loss",
                "forward_loss",
                "weighted_response_loss",
                "response_raw_mse",
            ),
        )
        self.writer.writeheader()
        self.record(0, 0, 0, 0, 0)

    def write(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        temp.replace(self.path)

    def record(
        self,
        iteration,
        steps,
        wm_updates,
        policy_updates,
        imagined_steps,
        wm_loss=None,
        response_loss=None,
        policy_loss=None,
        forward_loss=None,
        weighted_response_loss=None,
        response_raw_mse=None,
    ):
        self.data.update(
            env_steps=steps, wm_updates=wm_updates, policy_updates=policy_updates
        )
        self.writer.writerow(
            dict(
                iteration=iteration,
                env_steps=steps,
                wm_updates=wm_updates,
                policy_updates=policy_updates,
                imagined_steps=imagined_steps,
                elapsed_s=time.time() - self.data["started_unix"],
                wm_loss=wm_loss,
                response_loss=response_loss,
                policy_loss=policy_loss,
                forward_loss=forward_loss,
                weighted_response_loss=weighted_response_loss,
                response_raw_mse=response_raw_mse,
            )
        )
        self.metrics.flush()

    def finish(self, completed=True):
        self.data.update(
            status="completed" if completed else "truncated", finished_unix=time.time()
        )
        self.write()
        with (self.root / "checkpoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=(
                    "path",
                    "sha256",
                    "env_steps",
                    "iteration",
                    "contains_world_model",
                    "contains_critic",
                ),
            )
            w.writeheader()
            for p in sorted(self.root.glob("*.pt")):
                ck = torch.load(p, map_location="cpu", weights_only=False)
                w.writerow(
                    dict(
                        path=p.name,
                        sha256=sha256(p),
                        env_steps=ck.get("sim_steps", 0),
                        iteration=ck.get("iteration", 0),
                        contains_world_model="wm" in ck,
                        contains_critic="critic" in ck,
                    )
                )

    def interrupted(self):
        if self.data["status"] == "running":
            self.data.update(status="interrupted_or_failed", finished_unix=time.time())
            self.write()
