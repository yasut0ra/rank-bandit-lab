import unittest

from rank_bandit_lab import scenario_loader


class ScenarioLoaderTests(unittest.TestCase):
    def test_list_contains_known_scenarios(self) -> None:
        scenarios = scenario_loader.list_scenarios()
        expected = {"news_headlines", "ecommerce_longtail", "video_streaming", "education_catalog"}
        self.assertTrue(expected.issubset(set(scenarios)))

    def test_load_scenario_returns_documents(self) -> None:
        data = scenario_loader.load_scenario("news_headlines")
        self.assertIn("documents", data)
        self.assertGreater(len(data["documents"]), 0)

    def test_new_scenarios_have_metadata(self) -> None:
        for name in ("video_streaming", "education_catalog"):
            data = scenario_loader.load_scenario(name)
            self.assertIn("description", data)
            self.assertGreater(len(data["documents"]), 0)
