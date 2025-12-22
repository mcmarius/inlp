import pytest
from datetime import datetime
from analysis.date_utils import extract_date_intervals

def test_extract_simple_interval():
    text = "Perioada de control: 20.12.2025 - 22.12.2025"
    intervals = extract_date_intervals(text)
    assert len(intervals) == 1
    assert intervals[0] == (datetime(2025, 12, 20), datetime(2025, 12, 22))

def test_extract_reconstruct_year():
    text = "Comisarii au verificat piața între 20.12 și 22.12.2025"
    intervals = extract_date_intervals(text)
    assert len(intervals) == 1
    assert intervals[0] == (datetime(2025, 12, 20), datetime(2025, 12, 22))

def test_extract_reconstruct_year_text_format():
    text = "Controlul a avut loc între 20 decembrie și 22 decembrie 2025"
    intervals = extract_date_intervals(text)
    assert len(intervals) == 1
    assert intervals[0] == (datetime(2025, 12, 20), datetime(2025, 12, 22))

def test_extract_cross_year_boundary():
    text = "Intervalul 30.12 – 02.01.2025 a fost unul aglomerat"
    intervals = extract_date_intervals(text)
    assert len(intervals) == 1
    assert intervals[0] == (datetime(2024, 12, 30), datetime(2025, 1, 2))

def test_extract_multiple_intervals():
    text = "Primul: 01.01.2025-05.01.2025. Al doilea: 10.02 - 15.02.2025"
    intervals = extract_date_intervals(text)
    assert len(intervals) == 2
    assert intervals[0] == (datetime(2025, 1, 1), datetime(2025, 1, 5))
    assert intervals[1] == (datetime(2025, 2, 10), datetime(2025, 2, 15))

def test_extract_past_year():
    text = "În perioada 15.05 - 20.05.2023 s-au efectuat controale."
    intervals = extract_date_intervals(text)
    assert len(intervals) == 1
    assert intervals[0] == (datetime(2023, 5, 15), datetime(2023, 5, 20))
