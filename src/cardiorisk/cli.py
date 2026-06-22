"""Command-line interface (``cardiorisk ...``).

Thin wrapper over the library: parses arguments, configures logging, and calls
into :mod:`cardiorisk.training` and the serving stack. The heavy lifting lives
in importable, testable modules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from cardiorisk import __version__

app = typer.Typer(
    add_completion=False,
    help="CardioRisk - heart disease risk prediction pipeline.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to YAML config.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"cardiorisk {__version__}")


@app.command()
def train(
    config: ConfigOpt = Path("configs/default.yaml"),
    verbose: VerboseOpt = False,
) -> None:
    """Train, evaluate, and persist the model + artifacts."""
    _setup_logging(verbose)
    from cardiorisk.config import load_settings
    from cardiorisk.training import run_training

    settings = load_settings(config)
    report = run_training(settings)

    typer.secho(f"\nBest model: {report.best_model}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Decision threshold (recall-tuned): {report.threshold:.3f}")
    m = report.test_metrics_tuned
    typer.echo(
        "Test  | ROC-AUC {roc_auc:.3f}  PR-AUC {pr_auc:.3f}  "
        "recall {recall:.3f}  precision {precision:.3f}  acc {accuracy:.3f}".format(**m)
    )
    typer.echo(f"Top predictors: {', '.join(report.top_features)}")
    typer.echo(f"Artifacts written to: {report.artifacts_dir}/")


@app.command()
def predict(
    input_json: Annotated[Path, typer.Argument(help="JSON file of patient features.")],
    model_path: Annotated[Path, typer.Option(help="Path to model bundle.")] = Path(
        "artifacts/model.joblib"
    ),
    verbose: VerboseOpt = False,
) -> None:
    """Predict risk for a single patient described in a JSON file."""
    _setup_logging(verbose)
    from cardiorisk.api import PatientFeatures
    from cardiorisk.persistence import load_bundle

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    patient = PatientFeatures.model_validate(payload)
    bundle = load_bundle(model_path)
    probability = float(bundle.predict_proba(patient.to_frame())[0])
    at_risk = probability >= bundle.threshold
    typer.echo(
        json.dumps(
            {
                "probability": round(probability, 4),
                "at_risk": at_risk,
                "threshold": round(bundle.threshold, 4),
                "model_name": bundle.model_name,
            },
            indent=2,
        )
    )


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Launch the FastAPI prediction service with uvicorn."""
    import uvicorn

    uvicorn.run("cardiorisk.api:app", host=host, port=port, factory=False)


if __name__ == "__main__":
    app()
