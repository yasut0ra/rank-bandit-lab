import unittest

from rank_bandit_lab import scenario_loader


class ScenarioLoaderTests(unittest.TestCase):
    def test_list_contains_known_scenarios(self) -> None:
        scenarios = scenario_loader.list_scenarios()
        self.assertIn("news_headlines", scenarios)

    def test_load_scenario_returns_documents(self) -> None:
        data = scenario_loader.load_scenario("news_headlines")
        self.assertIn("documents", data)
        self.assertGreater(len(data["documents"]), 0)
