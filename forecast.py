#!/usr/bin/env python3
"""
Photographer Forecast (Australia) — sunrise/sunset colour, fog, storm/drama, plus tide times.

Data sources (free, no API key):
- Open-Meteo Weather Forecast API: https://api.open-meteo.com/v1/forecast
- Open-Meteo Marine API (sea level height incl. tides): https://marine-api.open-meteo.com/v1/marine

SSL note (macOS):
- If your Python install can't verify certificates, this script uses certifi's CA bundle.
  Install:  pip install certifi
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import sys
import textwrap
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# --- SSL / CA bundle patch (certifi) ---
import ssl

try:
    import certifi
except ImportError as e:
    raise SystemExit(
        "Missing dependency: certifi\n"
        "Install it with:\n"
        "  pip install certifi\n"
        "Then re-run the script."
    ) from e


# -----------------------------
# City list (Australia) — extend freely
# -----------------------------
@dataclasses.dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    tz: str
    coastal: bool = True


CITIES: List[City] = [
    City("Gold Coast", -28.0167, 153.4000, "Australia/Brisbane", coastal=True),
    City("Brisbane", -27.4698, 153.0251, "Australia/Brisbane", coastal=True),
    City("Sydney", -33.8688, 151.2093, "Australia/Sydney", coastal=True),
    City("Newcastle", -32.9283, 151.7817, "Australia/Sydney", coastal=True),
    City("Wollongong", -34.4278, 150.8931, "Australia/Sydney", coastal=True),
    City("Melbourne", -37.8136, 144.9631, "Australia/Melbourne", coastal=True),
    City("Geelong", -38.1499, 144.3617, "Australia/Melbourne", coastal=True),
    City("Hobart", -42.8821, 147.3272, "Australia/Hobart", coastal=True),
    City("Adelaide", -34.9285, 138.6007, "Australia/Adelaide", coastal=True),
    City("Perth", -31.9523, 115.8613, "Australia/Perth", coastal=True),
    City("Darwin", -12.4634, 130.8456, "Australia/Darwin", coastal=True),
    City("Cairns", -16.9186, 145.7781, "Australia/Brisbane", coastal=True),
    City("Townsville", -19.2589, 146.8169, "Australia/Brisbane", coastal=True),
    City("Byron Bay", -28.6474, 153.6020, "Australia/Sydney", coastal=True),
    City("Canberra", -35.2809, 149.1300, "Australia/Sydney", coastal=False),
]


# -----------------------------
# Helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def parse_local_iso(ts: str) -> dt.datetime:
    # Open-Meteo returns local-time ISO strings without offset when timezone is provided.
    return dt.datetime.fromisoformat(ts)


def date_key(d: dt.datetime) -> str:
    return d.date().isoformat()


def http_get_json(url: str, timeout: int = 20) -> Dict[str, Any]:
    """
    HTTPS GET JSON using certifi CA bundle (fixes macOS CERTIFICATE_VERIFY_FAILED cases).
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "photo-forecast/1.1 (Python urllib)",
            "Accept": "application/json",
        },
    )
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


# -----------------------------
# Simple file cache (optional but practical)
# -----------------------------
def cache_dir() -> str:
    base = os.path.join(os.path.expanduser("~"), ".photo_forecast_cache")
    os.makedirs(base, exist_ok=True)
    return base


def cache_path(key: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
    return os.path.join(cache_dir(), safe + ".json")


def cached_get(url: str, cache_key: str, max_age_seconds: int = 900) -> Dict[str, Any]:
    path = cache_path(cache_key)
    now = time.time()

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            ts = float(obj.get("_cached_at", 0))
            if now - ts <= max_age_seconds and "data" in obj:
                return obj["data"]
        except Exception:
            pass  # cache miss

    data = http_get_json(url)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"_cached_at": now, "data": data}, f)
    except Exception:
        pass
    return data


# -----------------------------
# Open-Meteo clients
# -----------------------------
def build_weather_url(city: City, days: int) -> str:
    base = "https://api.open-meteo.com/v1/forecast"
    hourly = ",".join(
        [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation_probability",
            "precipitation",
            "rain",
            "showers",
            "weather_code",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "visibility",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
            "surface_pressure",
            "cape",
            "is_day",
        ]
    )
    daily = ",".join(["sunrise", "sunset"])

    params = {
        "latitude": f"{city.lat:.6f}",
        "longitude": f"{city.lon:.6f}",
        "timezone": city.tz,
        "forecast_days": str(days),
        "hourly": hourly,
        "daily": daily,
        "wind_speed_unit": "kmh",
    }
    return base + "?" + urllib.parse.urlencode(params)


def build_marine_url(city: City, days: int) -> str:
    base = "https://marine-api.open-meteo.com/v1/marine"
    hourly = ",".join(
        [
            "sea_level_height_msl",  # includes tides
            "wave_height",
            "wave_period",
            "wave_direction",
        ]
    )
    params = {
        "latitude": f"{city.lat:.6f}",
        "longitude": f"{city.lon:.6f}",
        "timezone": city.tz,
        "forecast_days": str(min(days, 8)),
        "hourly": hourly,
        "length_unit": "metric",
        "cell_selection": "sea",
    }
    return base + "?" + urllib.parse.urlencode(params)


def fetch_forecasts(city: City, days: int, cache_seconds: int = 900) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    w_url = build_weather_url(city, days)
    w = cached_get(w_url, f"weather_{city.name}_{days}", max_age_seconds=cache_seconds)

    m = None
    if city.coastal:
        m_url = build_marine_url(city, days)
        try:
            m = cached_get(m_url, f"marine_{city.name}_{days}", max_age_seconds=cache_seconds)
        except Exception:
            m = None
    return w, m


# -----------------------------
# Scoring logic (photography-centric heuristics)
# -----------------------------
@dataclasses.dataclass
class Opportunity:
    kind: str            # "SUNRISE", "SUNSET", "FOG", "STORM"
    when: dt.datetime
    score: float
    summary: str
    details: Dict[str, Any]


def score_colour_window(row: Dict[str, float], prev_row: Optional[Dict[str, float]] = None) -> float:
    total = row.get("cloud_cover", math.nan)
    low = row.get("cloud_cover_low", math.nan)
    mid = row.get("cloud_cover_mid", math.nan)
    high = row.get("cloud_cover_high", math.nan)

    precip = row.get("precipitation", 0.0) or 0.0
    pprob = row.get("precipitation_probability", 0.0) or 0.0
    vis_m = row.get("visibility", 0.0) or 0.0
    gust = row.get("wind_gusts_10m", 0.0) or 0.0

    score = 0.0

    # Total cloud sweet spot ~ 25-70
    if not math.isnan(total):
        if total < 10:
            score += lerp(5, 15, clamp(total / 10, 0, 1))
        elif total <= 70:
            score += lerp(20, 40, (total - 10) / 60)
        else:
            score += lerp(40, 10, clamp((total - 70) / 30, 0, 1))

    # Prefer mid/high clouds
    if not math.isnan(high):
        score += lerp(0, 25, clamp(1 - abs(high - 40) / 40, 0, 1))
    if not math.isnan(mid):
        score += lerp(0, 15, clamp(1 - abs(mid - 35) / 45, 0, 1))

    # Penalise low cloud strongly
    if not math.isnan(low):
        score -= lerp(0, 35, clamp(low / 80, 0, 1))

    # Rain / precip probability penalties
    score -= lerp(0, 40, clamp(precip / 2.0, 0, 1))
    score -= lerp(0, 25, clamp(pprob / 80.0, 0, 1))

    # Visibility bonus
    vis_km = vis_m / 1000.0
    score += lerp(0, 15, clamp((vis_km - 6) / 14, 0, 1))

    # Wind gust penalty
    score -= lerp(0, 12, clamp((gust - 35) / 40, 0, 1))

    # Clearing-after-rain bonus
    if prev_row:
        prev_rain = (prev_row.get("rain", 0.0) or 0.0) + (prev_row.get("showers", 0.0) or 0.0)
        curr_rain = (row.get("rain", 0.0) or 0.0) + (row.get("showers", 0.0) or 0.0)
        if prev_rain > 0.5 and curr_rain < 0.2 and (row.get("cloud_cover", 0.0) or 0.0) > 15:
            score += 12

    return clamp(score, 0, 100)


def score_fog(row: Dict[str, float]) -> float:
    vis_m = row.get("visibility", 99999.0) or 99999.0
    rh = row.get("relative_humidity_2m", 0.0) or 0.0
    t = row.get("temperature_2m", math.nan)
    dp = row.get("dew_point_2m", math.nan)
    wind = row.get("wind_speed_10m", 0.0) or 0.0

    spread = None
    if not math.isnan(t) and not math.isnan(dp):
        spread = abs(t - dp)

    score = 0.0

    if vis_m <= 200:
        score += 60
    elif vis_m <= 1000:
        score += lerp(60, 35, (vis_m - 200) / 800)
    elif vis_m <= 3000:
        score += lerp(35, 10, (vis_m - 1000) / 2000)

    score += lerp(0, 20, clamp((rh - 90) / 10, 0, 1))

    if spread is not None:
        score += lerp(0, 25, clamp((1.5 - spread) / 1.5, 0, 1))

    score += lerp(0, 10, clamp((12 - wind) / 12, 0, 1))

    return clamp(score, 0, 100)


def score_storm(row: Dict[str, float]) -> float:
    cape = row.get("cape", 0.0) or 0.0
    pprob = row.get("precipitation_probability", 0.0) or 0.0
    gust = row.get("wind_gusts_10m", 0.0) or 0.0
    wcode = int(row.get("weather_code", 0) or 0)
    precip = row.get("precipitation", 0.0) or 0.0
    pressure = row.get("surface_pressure", 0.0) or 0.0

    score = 0.0

    score += lerp(0, 30, clamp(cape / 1500.0, 0, 1))
    score += lerp(0, 25, clamp(pprob / 80.0, 0, 1))
    score += lerp(0, 25, clamp((gust - 25) / 55.0, 0, 1))
    score += lerp(0, 15, clamp(precip / 4.0, 0, 1))

    if wcode in (95, 96, 99):
        score += 35
    elif wcode in (80, 81, 82, 61, 63, 65):
        score += 15

    if pressure > 0:
        score += lerp(0, 10, clamp((1015 - pressure) / 20.0, 0, 1))

    return clamp(score, 0, 100)


# -----------------------------
# Tide extraction from sea_level_height_msl
# -----------------------------
@dataclasses.dataclass
class TideEvent:
    kind: str  # "HIGH" or "LOW"
    when: dt.datetime
    height_m: float


def derive_tide_events(marine: Dict[str, Any]) -> List[TideEvent]:
    hourly = marine.get("hourly") or {}
    times = hourly.get("time") or []
    heights = hourly.get("sea_level_height_msl") or []

    if not times or not heights or len(times) != len(heights):
        return []

    dtimes = [parse_local_iso(t) for t in times]
    vals = [float(v) for v in heights]

    events: List[TideEvent] = []
    for i in range(1, len(vals) - 1):
        a, b, c = vals[i - 1], vals[i], vals[i + 1]
        if b > a and b > c:
            events.append(TideEvent("HIGH", dtimes[i], b))
        elif b < a and b < c:
            events.append(TideEvent("LOW", dtimes[i], b))

    # De-noise: keep events at least ~4 hours apart
    filtered: List[TideEvent] = []
    min_sep = dt.timedelta(hours=4)
    for ev in events:
        if not filtered or (ev.when - filtered[-1].when) >= min_sep:
            filtered.append(ev)

    return filtered


def nearest_tide(events: List[TideEvent], t: dt.datetime, within_hours: float = 2.5) -> Optional[TideEvent]:
    if not events:
        return None
    best = min(events, key=lambda e: abs((e.when - t).total_seconds()))
    if abs((best.when - t).total_seconds()) <= within_hours * 3600:
        return best
    return None


# -----------------------------
# Forecast shaping
# -----------------------------
def to_hourly_rows(weather: Dict[str, Any]) -> List[Tuple[dt.datetime, Dict[str, float]]]:
    hourly = weather.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return []

    keys = [k for k in hourly.keys() if k != "time"]
    rows = []
    for i, ts in enumerate(times):
        row: Dict[str, float] = {}
        for k in keys:
            arr = hourly.get(k) or []
            if i < len(arr):
                v = arr[i]
                try:
                    row[k] = float(v) if v is not None else math.nan
                except Exception:
                    row[k] = math.nan
        rows.append((parse_local_iso(ts), row))
    return rows


def daily_sun_times(weather: Dict[str, Any]) -> Dict[str, Dict[str, dt.datetime]]:
    daily = weather.get("daily") or {}
    days = daily.get("time") or []
    sunrise = daily.get("sunrise") or []
    sunset = daily.get("sunset") or []
    out: Dict[str, Dict[str, dt.datetime]] = {}

    for i, d in enumerate(days):
        if i < len(sunrise) and i < len(sunset):
            out[d] = {
                "sunrise": parse_local_iso(sunrise[i]),
                "sunset": parse_local_iso(sunset[i]),
            }
    return out


def pick_best_in_window(
    hourly_rows: List[Tuple[dt.datetime, Dict[str, float]]],
    center: dt.datetime,
    window_minutes: int,
    scorer,
) -> Optional[Tuple[dt.datetime, float, Dict[str, float], Optional[Dict[str, float]]]]:
    start = center - dt.timedelta(minutes=window_minutes)
    end = center + dt.timedelta(minutes=window_minutes)

    best = None
    for idx, (t, row) in enumerate(hourly_rows):
        if start <= t <= end:
            prev_row = hourly_rows[idx - 1][1] if idx - 1 >= 0 else None
            s = scorer(row, prev_row) if scorer == score_colour_window else scorer(row)
            if best is None or s > best[1]:
                best = (t, s, row, prev_row)
    return best


def build_opportunities(city: City, weather: Dict[str, Any], marine: Optional[Dict[str, Any]], days: int) -> List[Opportunity]:
    """
    Build photographer-friendly opportunities with plain-English reasons in every recommendation.
    Requires these helper functions to exist (paste them above this function):
      - why_sun_colour(row, prev_row)
      - why_fog(row)
      - why_storm(row)
    """
    hourly_rows = to_hourly_rows(weather)
    sun = daily_sun_times(weather)

    tide_events: List[TideEvent] = derive_tide_events(marine) if marine else []

    # Group rows by date
    rows_by_day: Dict[str, List[Tuple[dt.datetime, Dict[str, float]]]] = defaultdict(list)
    for t, row in hourly_rows:
        rows_by_day[date_key(t)].append((t, row))

    opps: List[Opportunity] = []

    # --- Sunrise / Sunset colour opportunities (with "Why: ...") ---
    for d, s in sun.items():
        if d not in rows_by_day:
            continue

        rows = rows_by_day[d]

        # Sunrise
        sr_pick = pick_best_in_window(rows, s["sunrise"], window_minutes=75, scorer=score_colour_window)
        if sr_pick:
            t, score, row, prev_row = sr_pick
            tide = nearest_tide(tide_events, t)
            tide_txt = f" • {tide.kind} tide {tide.when.strftime('%H:%M')} ({tide.height_m:.2f}m)" if tide else ""
            why = why_sun_colour(row, prev_row)
            opps.append(
                Opportunity(
                    kind="SUNRISE",
                    when=t,
                    score=score,
                    summary=f"Sunrise colour potential{tide_txt} | {why}",
                    details={"sunrise": s["sunrise"].isoformat(), **row},
                )
            )

        # Sunset
        ss_pick = pick_best_in_window(rows, s["sunset"], window_minutes=75, scorer=score_colour_window)
        if ss_pick:
            t, score, row, prev_row = ss_pick
            tide = nearest_tide(tide_events, t)
            tide_txt = f" • {tide.kind} tide {tide.when.strftime('%H:%M')} ({tide.height_m:.2f}m)" if tide else ""
            why = why_sun_colour(row, prev_row)
            opps.append(
                Opportunity(
                    kind="SUNSET",
                    when=t,
                    score=score,
                    summary=f"Sunset colour potential{tide_txt} | {why}",
                    details={"sunset": s["sunset"].isoformat(), **row},
                )
            )

    # --- Fog candidates: early morning (02:00–10:00) and evening (18:00–23:00) ---
    for t, row in hourly_rows:
        hr = t.hour
        if (2 <= hr <= 10) or (18 <= hr <= 23):
            s = score_fog(row)
            if s >= 55:
                tide = nearest_tide(tide_events, t)
                tide_txt = f" • {tide.kind} tide {tide.when.strftime('%H:%M')} ({tide.height_m:.2f}m)" if tide else ""
                why = why_fog(row)
                opps.append(
                    Opportunity(
                        kind="FOG",
                        when=t,
                        score=s,
                        summary=f"Fog/mist potential{tide_txt} | {why}",
                        details=row,
                    )
                )

    # --- Storm/drama candidates: any time, thresholded ---
    for t, row in hourly_rows:
        s = score_storm(row)
        if s >= 60:
            tide = nearest_tide(tide_events, t)
            tide_txt = f" • {tide.kind} tide {tide.when.strftime('%H:%M')} ({tide.height_m:.2f}m)" if tide else ""
            why = why_storm(row)
            opps.append(
                Opportunity(
                    kind="STORM",
                    when=t,
                    score=s,
                    summary=f"Storm/drama potential{tide_txt} | {why}",
                    details=row,
                )
            )

    # Sort: higher score first, then earlier time
    opps.sort(key=lambda o: (-o.score, o.when))
    return opps

def why_sun_colour(row: Dict[str, float], prev_row: Optional[Dict[str, float]] = None) -> str:
    total = row.get("cloud_cover", math.nan)
    low = row.get("cloud_cover_low", math.nan)
    mid = row.get("cloud_cover_mid", math.nan)
    high = row.get("cloud_cover_high", math.nan)
    vis_km = (row.get("visibility", 0.0) or 0.0) / 1000.0
    precip = row.get("precipitation", 0.0) or 0.0
    pprob = row.get("precipitation_probability", 0.0) or 0.0

    parts = []

    # Cloud story
    if not math.isnan(total):
        if 25 <= total <= 70:
            parts.append(f"good cloud amount ({total:.0f}%) to catch colour")
        elif total < 25:
            parts.append(f"mostly clear ({total:.0f}%) — can be clean but less colour")
        else:
            parts.append(f"very cloudy ({total:.0f}%) — may block light")

    if not math.isnan(low):
        if low >= 60:
            parts.append(f"lots of LOW cloud ({low:.0f}%) — often kills sunrise/sunset colour")
        elif low >= 35:
            parts.append(f"some LOW cloud ({low:.0f}%) — could block horizon")
        else:
            parts.append(f"low cloud is limited ({low:.0f}%) — good for horizon glow")

    if not math.isnan(mid) or not math.isnan(high):
        mh = []
        if not math.isnan(mid):
            mh.append(f"mid {mid:.0f}%")
        if not math.isnan(high):
            mh.append(f"high {high:.0f}%")
        if mh:
            parts.append("colour-catching cloud layers (" + ", ".join(mh) + ")")

    # Visibility story
    if vis_km > 0:
        if vis_km >= 15:
            parts.append(f"clear air (visibility ~{vis_km:.0f} km)")
        elif vis_km >= 8:
            parts.append(f"okay clarity (visibility ~{vis_km:.0f} km)")
        else:
            parts.append(f"hazy air (visibility ~{vis_km:.0f} km)")

    # Rain / clearing story
    if precip >= 1.0 or pprob >= 70:
        parts.append("rain likely — could block light, but breaks can be dramatic")
    if prev_row:
        prev_rain = (prev_row.get("rain", 0.0) or 0.0) + (prev_row.get("showers", 0.0) or 0.0)
        curr_rain = (row.get("rain", 0.0) or 0.0) + (row.get("showers", 0.0) or 0.0)
        if prev_rain > 0.5 and curr_rain < 0.2:
            parts.append("clearing after rain — often boosts colour and contrast")

    return "Why: " + "; ".join(parts[:5])


def why_fog(row: Dict[str, float]) -> str:
    vis_m = row.get("visibility", 99999.0) or 99999.0
    rh = row.get("relative_humidity_2m", math.nan)
    t = row.get("temperature_2m", math.nan)
    dp = row.get("dew_point_2m", math.nan)
    wind = row.get("wind_speed_10m", math.nan)

    parts = []

    # Visibility headline
    if vis_m <= 200:
        parts.append("very low visibility (thick fog/mist likely)")
    elif vis_m <= 1000:
        parts.append("reduced visibility (mist/fog possible)")
    elif vis_m <= 3000:
        parts.append("some haze (light mist possible)")
    else:
        parts.append("visibility not especially low (fog less likely)")

    # RH explained
    if not math.isnan(rh):
        parts.append(f"RH {rh:.0f}% (Relative Humidity = how moisture-saturated the air is)")

    # Dew point spread explained
    if not math.isnan(t) and not math.isnan(dp):
        spread = abs(t - dp)
        parts.append(f"temp–dew point spread ~{spread:.1f}°C (closer = fog more likely)")

    # Wind explained
    if not math.isnan(wind):
        if wind <= 8:
            parts.append(f"light wind {wind:.0f} km/h (fog can hang around)")
        else:
            parts.append(f"wind {wind:.0f} km/h (may disperse fog)")

    return "Why: " + "; ".join(parts[:5])


def why_storm(row: Dict[str, float]) -> str:
    cape = row.get("cape", math.nan)
    pprob = row.get("precipitation_probability", math.nan)
    gust = row.get("wind_gusts_10m", math.nan)
    wcode = int(row.get("weather_code", 0) or 0)
    precip = row.get("precipitation", math.nan)

    parts = []

    # CAPE explained
    if not math.isnan(cape):
        if cape >= 1200:
            parts.append(f"CAPE {cape:.0f} (storm energy) — strong storm potential")
        elif cape >= 600:
            parts.append(f"CAPE {cape:.0f} (storm energy) — some storm potential")
        else:
            parts.append(f"CAPE {cape:.0f} (storm energy) — limited storm potential")

    # Rain chance / amount
    if not math.isnan(pprob):
        parts.append(f"rain chance {pprob:.0f}%")
    if not math.isnan(precip) and precip > 0:
        parts.append(f"precip ~{precip:.1f} mm/hr (can add drama / curtains of rain)")

    # Gusts
    if not math.isnan(gust):
        if gust >= 45:
            parts.append(f"strong gusts {gust:.0f} km/h (dramatic clouds, but tripod needed)")
        elif gust >= 30:
            parts.append(f"gusts {gust:.0f} km/h (movement/drama possible)")
        else:
            parts.append(f"gusts {gust:.0f} km/h")

    # Weather codes hint
    if wcode in (95, 96, 99):
        parts.append("thunderstorm code detected")
    elif wcode in (80, 81, 82):
        parts.append("showery code detected")

    return "Why: " + "; ".join(parts[:5])

def format_opportunity(o: Opportunity) -> str:
    when = o.when.strftime("%a %d %b %H:%M")
    line = f"{when}  [{o.kind:<7}]  score {o.score:5.1f}  {o.summary}"
    return textwrap.fill(line, width=110, subsequent_indent=" " * 34)


def summarise_tides(city: City, marine: Optional[Dict[str, Any]]) -> str:
    if not city.coastal:
        return "Tides: (not coastal city — skipped)\n"
    if not marine:
        return "Tides: (marine data unavailable)\n"

    events = derive_tide_events(marine)
    if not events:
        return "Tides: (no tide events detected)\n"

    by_day: Dict[str, List[TideEvent]] = defaultdict(list)
    for ev in events:
        by_day[date_key(ev.when)].append(ev)

    lines = ["Tides (approx; derived from sea level height incl. tides):"]
    for d in sorted(by_day.keys()):
        evs = by_day[d]
        highs = [e for e in evs if e.kind == "HIGH"][:2]
        lows = [e for e in evs if e.kind == "LOW"][:2]
        parts = []
        for e in highs + lows:
            parts.append(f"{e.kind} {e.when.strftime('%H:%M')} {e.height_m:.2f}m")
        if parts:
            lines.append(f"  {d}: " + " | ".join(parts))
    return "\n".join(lines) + "\n"


# -----------------------------
# CLI
# -----------------------------
def find_city(name: str) -> Optional[City]:
    name_norm = name.strip().lower()
    for c in CITIES:
        if c.name.lower() == name_norm:
            return c
    for c in CITIES:
        if name_norm in c.name.lower():
            return c
    return None


def list_cities() -> str:
    lines = ["Available cities:"]
    for c in CITIES:
        coast = "coastal" if c.coastal else "inland"
        lines.append(f"  - {c.name} ({coast}, {c.tz})")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="photo_forecast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Photographer Forecast (Australia)
            - Scores sunrise/sunset colour potential, fog, and storm/drama.
            - Includes approximate tide times (high/low) for coastal cities via Open-Meteo Marine API.
            """
        ),
    )
    parser.add_argument("--city", default="Gold Coast", help="City name (default: Gold Coast). Use --list to see options.")
    parser.add_argument("--days", type=int, default=10, help="Forecast days (default: 3)")
    parser.add_argument("--top", type=int, default=12, help="Show top N opportunities (default: 12)")
    parser.add_argument("--list", action="store_true", help="List available cities and exit")
    parser.add_argument("--json", action="store_true", help="Output opportunities as JSON")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching (always fetch fresh)")

    args = parser.parse_args()

    if args.list:
        print(list_cities())
        return 0

    city = find_city(args.city)
    if not city:
        print(f"Unknown city: {args.city}\n")
        print(list_cities())
        return 2

    days = int(clamp(args.days, 1, 16))
    cache_seconds = 0 if args.no_cache else 900

    weather, marine = fetch_forecasts(city, days, cache_seconds=cache_seconds)

    if "hourly" not in weather or "time" not in (weather.get("hourly") or {}):
        print("Weather data missing expected fields. Raw response (truncated):\n")
        print(json.dumps(weather, indent=2)[:2000])
        return 1

    opps = build_opportunities(city, weather, marine, days)

    header = f"{city.name} — Photographer Forecast ({days} day(s))"
    print(header)
    print("=" * len(header))

    print(summarise_tides(city, marine))

    if args.json:
        out = [
            {
                "kind": o.kind,
                "when_local": o.when.isoformat(),
                "score": round(o.score, 1),
                "summary": o.summary,
                "details": o.details,
            }
            for o in opps[: args.top]
        ]
        print(json.dumps({"city": city.name, "timezone": city.tz, "opportunities": out}, indent=2))
        return 0

    if not opps:
        print("No strong opportunities detected with current thresholds.")
        print("Tip: run with --json to inspect details, increase --days, or relax thresholds in code.")
        return 0

    print(f"Top {min(args.top, len(opps))} opportunities:")
    for o in opps[: args.top]:
        print("  " + format_opportunity(o))

    print("\nLegend:")
    print("  SUNRISE/SUNSET: colourful sky potential (cloud layers + clearing + visibility)")
    print("  FOG: low visibility + high RH + small temp/dewpoint spread + light wind")
    print("  STORM: instability (CAPE) + precip probability + gusts + thunder/rain codes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
