import pytest

from openmv_cam.event_diagnostics import (
    DurationSamples,
    EventDiagnosticCounters,
)


def test_duration_samples_are_bounded_and_report_percentiles():
    samples = DurationSamples(capacity=3)
    for value in (0.001, 0.002, 0.003, 0.004):
        samples.add_seconds(value)
    p50, p95, maximum = samples.summary_ms()
    assert p50 == pytest.approx(3.0)
    assert p95 == pytest.approx(3.9)
    assert maximum == pytest.approx(4.0)


def test_diagnostic_counters_track_skips_drops_and_anchor_state():
    counters = EventDiagnosticCounters()
    counters.timer_ticks += 2
    counters.messages_published += 1
    counters.skipped_no_new_events += 1
    counters.packets_dropped_by_cap += 3
    counters.events_dropped_by_cap += 40
    counters.anchors_advanced += 1
    assert counters.timer_ticks == 2
    assert counters.messages_published == 1
    assert counters.skipped_no_new_events == 1
    assert counters.packets_dropped_by_cap == 3
    assert counters.events_dropped_by_cap == 40
    assert counters.anchors_advanced == 1
