from experiment.optimization.sensitivity import evaluate_gate, sweep_gates


def _rows():
    return [
        {"confidence_source": "initial_runtime_evidence", "confidence": "0.8", "uniqueness_margin": "0.10", "disparity_error": "0.2"},
        {"confidence_source": "initial_runtime_evidence", "confidence": "0.3", "uniqueness_margin": "0.02", "disparity_error": "12.0"},
        {"confidence_source": "legacy_or_temporal", "confidence": "0", "uniqueness_margin": "", "disparity_error": ""},
    ]


def test_gate_reports_coverage_and_catastrophic_error_rate():
    result = evaluate_gate(_rows(), uniqueness_margin=0.01, confidence_threshold=0.25)
    assert result["accepted"] == 2
    assert result["coverage"] == 2 / 3
    assert result["cer_10"] == 0.5


def test_sweep_can_remove_catastrophe_at_cost_of_coverage():
    rows = sweep_gates(_rows(), uniqueness_margins=[0.01], confidence_thresholds=[0.25, 0.5])
    assert rows[1]["accepted"] == 1
    assert rows[1]["cer_10"] == 0.0
