"""Duong dan chung — moi model co thu muc rieng trong results/ va models_saved/."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelPaths:
    """Duong dan output cho tung thuat toan: lstm | stgcn."""

    def __init__(self, name: str):
        if name not in ("lstm", "stgcn"):
            raise ValueError(f"Model khong hop le: {name}")
        self.name = name
        self.results_dir = PROJECT_ROOT / "results" / name
        self.weights_dir = PROJECT_ROOT / "models_saved" / name

    @property
    def model_path(self) -> Path:
        return self.weights_dir / "best.pt"

    @property
    def training_log(self) -> Path:
        return self.results_dir / "training_log.csv"

    @property
    def training_history(self) -> Path:
        return self.results_dir / "training_history.png"

    @property
    def test_metrics(self) -> Path:
        return self.results_dir / "test_metrics.json"

    @property
    def confusion_matrix(self) -> Path:
        return self.results_dir / "confusion_matrix.png"

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
