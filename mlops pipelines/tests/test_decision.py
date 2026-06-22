import os
import sys
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "components", "common"),
)

from decision import derive_decision  # noqa: E402


def test_risk_low():
    assert derive_decision(0.0, 0.0, 0.1)[0] == "low"

def test_risk_medium():
    assert derive_decision(0.0, 0.0, 0.5)[0] == "medium"

def test_risk_high():
    assert derive_decision(0.0, 0.0, 0.9)[0] == "high"

def test_action_dispatch():
    assert derive_decision(0.0, 0.8, 0.9)[1] == "dispatch_technician"

def test_action_monitor():
    assert derive_decision(0.8, 0.0, 0.5)[1] == "monitor"

def test_action_none():
    assert derive_decision(0.1, 0.1, 0.2)[1] == "no_action"

def test_priority_1():
    assert derive_decision(0.0, 0.9, 0.9)[2] == 1

def test_priority_2():
    assert derive_decision(0.0, 0.0, 0.6)[2] == 2

def test_priority_3():
    assert derive_decision(0.0, 0.0, 0.2)[2] == 3