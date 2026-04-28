from src.agent.ablation import AblationMode


class TestAblationMode:
    def test_enum_values(self):
        assert AblationMode.FULL == "full"
        assert AblationMode.NO_VERIFIER == "no-verifier"
        assert AblationMode.NO_SYNTHESIZER == "no-synthesizer"
        assert AblationMode.POC == "poc"

    def test_enum_from_string(self):
        assert AblationMode("full") == AblationMode.FULL
        assert AblationMode("no-verifier") == AblationMode.NO_VERIFIER
        assert AblationMode("no-synthesizer") == AblationMode.NO_SYNTHESIZER
        assert AblationMode("poc") == AblationMode.POC

    def test_all_modes_listed(self):
        assert len(AblationMode) == 4
