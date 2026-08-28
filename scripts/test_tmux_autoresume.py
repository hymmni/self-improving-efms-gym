#!/usr/bin/env python3
"""tmux_autoresume.py 순수 로직(파싱·감지) 테스트. tmux 없이도 돈다."""

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tmux_autoresume as t


# ── strip_ansi ──────────────────────────────────────────────

def test_strip_ansi_removes_color_codes():
    assert t.strip_ansi("\x1b[33mhello\x1b[0m") == "hello"


def test_strip_ansi_noop_on_plain():
    assert t.strip_ansi("plain text") == "plain text"


# ── LIMIT_HINT 감지 ─────────────────────────────────────────

def test_limit_hint_detects_reached():
    assert t.LIMIT_HINT.search("5-hour limit reached . resets 3pm")


def test_limit_hint_ignores_normal_text():
    assert not t.LIMIT_HINT.search("그냥 평범한 작업 화면입니다")


# ── parse_reset: 시각만 ─────────────────────────────────────

def test_parse_reset_pm_hour_only():
    r = t.parse_reset("resets 3pm")
    assert r is not None and r.hour == 15 and r.minute == 0


def test_parse_reset_with_minutes():
    r = t.parse_reset("resets 3:30pm")
    assert r is not None and r.hour == 15 and r.minute == 30


def test_parse_reset_am():
    r = t.parse_reset("resets at 11 am")
    assert r is not None and r.hour == 11


def test_parse_reset_12am_is_midnight():
    r = t.parse_reset("resets 12am")
    assert r is not None and r.hour == 0


def test_parse_reset_12pm_is_noon():
    r = t.parse_reset("resets 12pm")
    assert r is not None and r.hour == 12


def test_parse_reset_time_only_is_future():
    # 시각만 주어지면 항상 미래(오늘 남았으면 오늘, 지났으면 내일)여야 한다.
    r = t.parse_reset("resets 3pm")
    assert r is not None and r > datetime.datetime.now()


# ── parse_reset: 날짜 포함 ──────────────────────────────────

def test_parse_reset_with_date():
    r = t.parse_reset("Usage limit reached — resets Jun 10, 7:10pm (Asia/Seoul)")
    assert r is not None and r.month == 6 and r.day == 10 and r.hour == 19 and r.minute == 10


def test_parse_reset_with_ansi():
    r = t.parse_reset("\x1b[33m5-hour limit reached . resets 9pm\x1b[0m")
    assert r is not None and r.hour == 21


def test_parse_reset_returns_none_on_garbage():
    assert t.parse_reset("아무 의미 없는 텍스트") is None


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
