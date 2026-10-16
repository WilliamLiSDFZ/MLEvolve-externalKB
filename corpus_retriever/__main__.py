"""Run with: python -m corpus_retriever exp_id=... desc_file=... data_dir=..."""
import logging
import sys
from pathlib import Path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in args):
        print(__doc__)
        print("Also set abstract_index_path and export TYPESAFE_API_KEY / LLM_API_KEY.")
        print("See corpus_retriever/README.md. Re-running replaces cache/techinique/<exp_id>/.")
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from omegaconf import OmegaConf
        from config import Config
        from .retrieval import run_retrieval

        # load_cfg also creates an experiment workspace; this command only prepares papers.
        path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
        values = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(args))
        values.exp_name = values.exp_name or values.exp_id
        cfg = OmegaConf.merge(OmegaConf.structured(Config), values)
        manifest = run_retrieval(cfg)
        logging.getLogger("MLEvolve").info(
            "Corpus preparation %s: %s", manifest["status"], manifest["output_dir"])
        return 0 if manifest["status"] == "complete" else 2
    except Exception as exc:
        logging.getLogger("MLEvolve").error("Corpus preparation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
