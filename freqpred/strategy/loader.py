"""Strategy loader: resolve a strategy by built-in name or file path."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

from freqpred.strategy.base import IPredictionStrategy

# Map of built-in strategy names to their module paths.
_BUILTIN_STRATEGIES: dict[str, str] = {
    "ConservativeDefault": "freqpred.strategy.defaults.conservative",
    "PoliticsEdgeStrategy": "freqpred.strategy.defaults.politics",
    "TechNewsStrategy": "freqpred.strategy.defaults.tech",
    "FreshMarketStrategy": "freqpred.strategy.defaults.fresh_market",
    "DemoHarness": "freqpred.strategy.defaults.demo_harness",
}


def load_strategy(name_or_path: str) -> IPredictionStrategy:
    """Load and instantiate a strategy by name or file path.

    Supports:
    - Built-in strategy names (e.g. ``"ConservativeDefault"``)
    - Absolute or relative paths to a ``.py`` file containing exactly one
      ``IPredictionStrategy`` subclass (e.g. ``"strategies/my_strat.py"``)

    Args:
        name_or_path: Built-in strategy name or path to a strategy file.

    Returns:
        An instantiated IPredictionStrategy.

    Raises:
        ValueError: If the name is unknown or the file contains no valid strategy.
        FileNotFoundError: If the path does not exist.
    """
    path = Path(name_or_path)

    if path.suffix == ".py" or path.exists():
        return _load_from_file(path)

    if name_or_path in _BUILTIN_STRATEGIES:
        return _load_builtin(name_or_path)

    raise ValueError(
        f"Unknown strategy {name_or_path!r}. "
        f"Use a built-in name ({', '.join(_BUILTIN_STRATEGIES)}) "
        f"or a path to a .py file."
    )


def _load_builtin(name: str) -> IPredictionStrategy:
    module_path = _BUILTIN_STRATEGIES[name]
    module = importlib.import_module(module_path)
    cls = _find_strategy_class(module, name)
    return cls()


def _load_from_file(path: Path) -> IPredictionStrategy:
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    module_name = f"_freqpred_user_strategy_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    cls = _find_strategy_class(module)
    return cls()


def _find_strategy_class(
    module: object,
    preferred_name: str | None = None,
) -> type[IPredictionStrategy]:
    """Return the first concrete IPredictionStrategy subclass in *module*."""
    candidates: list[type[IPredictionStrategy]] = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if (
            issubclass(obj, IPredictionStrategy)
            and obj is not IPredictionStrategy
            and not inspect.isabstract(obj)
        )
    ]

    if not candidates:
        raise ValueError(
            f"No concrete IPredictionStrategy subclass found in {module.__name__!r}"
        )

    if preferred_name:
        for cls in candidates:
            if cls.__name__ == preferred_name:
                return cls

    return candidates[0]
