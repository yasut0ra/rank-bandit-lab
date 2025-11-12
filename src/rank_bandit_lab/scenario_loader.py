from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Dict, Iterable, List


SCENARIO_PACKAGE = "rank_bandit_lab.scenarios"


def list_scenarios() -> List[str]:
    contents = []
    for resource in resources.files(SCENARIO_PACKAGE).iterdir():
        if resource.name.endswith(".json"):
            contents.append(resource.name.removesuffix(".json"))
    contents.sort()
    return contents


def load_scenario(name: str) -> Dict:
    resource_name = f"{name}.json"
    try:
        data = resources.files(SCENARIO_PACKAGE).joinpath(resource_name).read_text()
    except (FileNotFoundError, AttributeError):
        raise ValueError(f"Scenario '{name}' not found.")
    return json.loads(data)
