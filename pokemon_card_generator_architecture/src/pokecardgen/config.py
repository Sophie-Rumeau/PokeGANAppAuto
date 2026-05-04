from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """Accès config.key en plus de config['key']."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[item] = value
        return value


def _to_dotdict(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, DotDict):
        return DotDict({k: _to_dotdict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_dotdict(v) for v in value]
    return value


def find_project_root(start: Path | None = None) -> Path:
    """
    Trouve la racine du projet sans dépendre du dossier courant.

    On part de ce fichier (`src/pokecardgen/config.py`) et on remonte jusqu'au
    dossier qui contient `pyproject.toml` ou `configs/default.yaml`.
    """
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "pokecardgen").exists():
            return parent
        if (parent / "configs" / "default.yaml").exists() and (parent / "src" / "pokecardgen").exists():
            return parent

    # Fallback pour la structure src/ classique : root/src/pokecardgen/config.py
    # config.py -> pokecardgen -> src -> root
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def load_config(path: str | Path | None = None) -> DotDict:
    """
    Charge la configuration sans utiliser le dossier courant.

    - `path=None` : utilise `configs/default.yaml` à la racine du projet.
    - `path="..."` relatif : résolu depuis la racine du projet.
    - `path` absolu : utilisé tel quel.
    """
    if path is None:
        if DEFAULT_CONFIG_PATH.exists():
            config_path = DEFAULT_CONFIG_PATH.resolve()
            text = config_path.read_text(encoding="utf-8")
        else:
            # Fallback utile si le package est installé sans dossier projet complet.
            config_path = None
            text = resources.files("pokecardgen.configs").joinpath("default.yaml").read_text(encoding="utf-8")
    else:
        config_path = _resolve_from_project(path)
        text = config_path.read_text(encoding="utf-8")

    raw = yaml.safe_load(text) or {}
    raw.setdefault("_meta", {})
    raw["_meta"]["project_root"] = str(PROJECT_ROOT)
    raw["_meta"]["config_path"] = str(config_path) if config_path else None
    raw["_meta"]["config_dir"] = str(config_path.parent) if config_path else None
    return _to_dotdict(raw)


def _is_prefixed_by_dir_name(path: Path, dir_path: Path) -> bool:
    return bool(path.parts) and path.parts[0].lower() == dir_path.name.lower()


@dataclass(frozen=True)
class RuntimePaths:
    """
    Chemins runtime entièrement relatifs à la racine du projet.

    Aucun chemin absolu n'est codé en dur. Les chemins relatifs du YAML sont
    toujours interprétés depuis PROJECT_ROOT.
    """

    project_root: Path
    config_dir: Path | None
    checkpoints_dir: Path
    data_dir: Path
    output_dir: Path

    def resolve(self, path: str | Path | None, *, base: Path | None = None) -> Path | None:
        if path is None:
            return None
        p = Path(path).expanduser()
        if p.is_absolute():
            return p.resolve()
        return ((base or self.project_root) / p).resolve()

    def model(self, path: str | Path) -> Path:
        """Résout un checkpoint relativement à `checkpoints/`."""
        p = Path(path).expanduser()
        if p.is_absolute():
            return p.resolve()
        if _is_prefixed_by_dir_name(p, self.checkpoints_dir):
            return (self.project_root / p).resolve()
        return (self.checkpoints_dir / p).resolve()

    def data_file(self, path: str | Path) -> Path:
        """Résout un fichier de données relativement à `data/`."""
        p = Path(path).expanduser()
        if p.is_absolute():
            return p.resolve()
        if _is_prefixed_by_dir_name(p, self.data_dir):
            return (self.project_root / p).resolve()
        return (self.data_dir / p).resolve()

    def output_file(self, path: str | Path) -> Path:
        """Résout un fichier de sortie relativement à `outputs/`."""
        p = Path(path).expanduser()
        if p.is_absolute():
            return p.resolve()
        if _is_prefixed_by_dir_name(p, self.output_dir):
            return (self.project_root / p).resolve()
        return (self.output_dir / p).resolve()


def _dir_from_config(config: DotDict, key: str, default_name: str) -> Path:
    cfg_value = config.get("paths", {}).get(key, default_name)
    p = Path(cfg_value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def build_runtime_paths(config: DotDict) -> RuntimePaths:
    meta = config.get("_meta", {})
    config_dir = Path(meta["config_dir"]).resolve() if meta.get("config_dir") else None

    return RuntimePaths(
        project_root=PROJECT_ROOT,
        config_dir=config_dir,
        checkpoints_dir=_dir_from_config(config, "checkpoints_dir", "checkpoints"),
        data_dir=_dir_from_config(config, "data_dir", "data"),
        output_dir=_dir_from_config(config, "output_dir", "outputs"),
    )


# Compatibilité avec les anciens plugins éventuels.
def resolve_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path(project_root).resolve() if project_root else PROJECT_ROOT / p).resolve()
