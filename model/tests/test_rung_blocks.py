"""Added 7 Oct 2026 (lead): Deviation 3 of the D8 note -- controls in every rung incl. R0; arm A71 keeps
the placement written at the freeze (controls from R1); HARCJ is the pure HAR-CJ reference outside the MCS."""
from model.config import EngineConfig, MCS_EXCLUDE, rung_blocks_for


def test_main_ladder_has_controls_in_r0():
    rb = rung_blocks_for(EngineConfig.for_spec("A00"))
    assert rb["R0"] == ("har", "control")
    for r in ("R1", "R2", "R3"):
        assert "control" in rb[r]
    assert rb["R1"][-1] == "attention" and rb["R2"][-1] == "polarity" and rb["R3"][-1] == "emotion"
    assert rb["HARCJ"] == ("har",) and "HARCJ" in MCS_EXCLUDE


def test_arm_a71_controls_from_r1():
    rb = rung_blocks_for(EngineConfig.for_spec("A71"))
    assert rb["R0"] == ("har",) and "control" in rb["R1"] and "attention" in rb["R1"]
