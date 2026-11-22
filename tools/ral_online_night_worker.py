"""Full fit/cal backend coverage in a separate process; frozen old checker unchanged."""
import argparse
from pathlib import Path

from ral_eight_hour import ROOT, atomic_json, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "motions", "params", "responses"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    import ral_backend_check as backend
    # The legacy checker is reusable with a different coverage count. Numerical
    # thresholds and all amplitudes stay unchanged, including previously failed rows.
    cfg = read(ROOT / "configs/ral/backend_check_v1.json")
    cfg.update(blocks_per_split=1000000, selection="All fit/cal blocks in lexical order",
               scope="Complete development splits; not locked test or formal equivalence")
    config = a.out.parent / (a.out.name+"_config.json")
    atomic_json(config, cfg)
    backend.CONFIG = config
    return backend.run_check(a)


if __name__ == "__main__":
    raise SystemExit(main())
