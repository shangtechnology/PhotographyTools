#!/usr/bin/env python3
"""
Photographer Weather Dashboard (Australia)
- Free weather + marine (tides) from Open-Meteo (no API key).
- Ranks opportunities: Sunrise colour, Sunset colour, Fog/Mist, Storm/Drama.
- Fixes UX issues: readable table, click-to-select (with fallback), focused graphs, at-a-glance best day.

Run:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install streamlit pandas plotly requests certifi
  streamlit run forecast_dashboard.py
"""

from __future__ import annotations

import math
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import certifi
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# -----------------------------
# Cities (Australia)
# -----------------------------
@dataclass(frozen=True)
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


def get_city(name: str) -> City:
    for c in CITIES:
        if c.name == name:
            return c
    return CITIES[0]


# -----------------------------
# API URLs (Open-Meteo)
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
        "length_unit": "metric",
        "cell_selection": "sea",
        "hourly": hourly,
    }
    return base + "?" + urllib.parse.urlencode(params)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_json(url: str) -> Dict[str, Any]:
    r = requests.get(
        url,
        timeout=25,
        verify=certifi.where(),  # avoids macOS CA issues
        headers={"User-Agent": "photo-dashboard/1.1"},
    )
    r.raise_for_status()
    return r.json()


# -----------------------------
# Dataframes
# -----------------------------
def hourly_to_df(weather: Dict[str, Any]) -> pd.DataFrame:
    h = weather.get("hourly", {}) or {}
    if "time" not in h:
        return pd.DataFrame()
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


def marine_to_df(marine: Dict[str, Any]) -> pd.DataFrame:
    h = (marine or {}).get("hourly", {}) or {}
    if not h or "time" not in h:
        return pd.DataFrame()
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


def daily_sun_df(weather: Dict[str, Any]) -> pd.DataFrame:
    d = weather.get("daily", {}) or {}
    if "time" not in d:
        return pd.DataFrame()
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"])
    df["sunrise"] = pd.to_datetime(df["sunrise"])
    df["sunset"] = pd.to_datetime(df["sunset"])
    return df.set_index("time").sort_index()


# -----------------------------
# Scoring helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def score_colour_window(row: pd.Series, prev_row: Optional[pd.Series] = None) -> float:
    total = float(row.get("cloud_cover", math.nan))
    low = float(row.get("cloud_cover_low", math.nan))
    mid = float(row.get("cloud_cover_mid", math.nan))
    high = float(row.get("cloud_cover_high", math.nan))

    precip = float(row.get("precipitation", 0.0) or 0.0)
    pprob = float(row.get("precipitation_probability", 0.0) or 0.0)
    vis_m = float(row.get("visibility", 0.0) or 0.0)
    gust = float(row.get("wind_gusts_10m", 0.0) or 0.0)

    score = 0.0

    # total cloud sweet spot
    if not math.isnan(total):
        if total < 10:
            score += lerp(5, 15, clamp(total / 10, 0, 1))
        elif total <= 70:
            score += lerp(20, 40, (total - 10) / 60)
        else:
            score += lerp(40, 10, clamp((total - 70) / 30, 0, 1))

    # mid/high helps colour
    if not math.isnan(high):
        score += lerp(0, 25, clamp(1 - abs(high - 40) / 40, 0, 1))
    if not math.isnan(mid):
        score += lerp(0, 15, clamp(1 - abs(mid - 35) / 45, 0, 1))

    # low cloud kills horizon glow
    if not math.isnan(low):
        score -= lerp(0, 35, clamp(low / 80, 0, 1))

    # rain penalties
    score -= lerp(0, 40, clamp(precip / 2.0, 0, 1))
    score -= lerp(0, 25, clamp(pprob / 80.0, 0, 1))

    # visibility bonus
    vis_km = vis_m / 1000.0
    score += lerp(0, 15, clamp((vis_km - 6) / 14, 0, 1))

    # gust penalty
    score -= lerp(0, 12, clamp((gust - 35) / 40, 0, 1))

    # clearing-after-rain bonus
    if prev_row is not None:
        prev_r = float(prev_row.get("rain", 0.0) or 0.0) + float(prev_row.get("showers", 0.0) or 0.0)
        curr_r = float(row.get("rain", 0.0) or 0.0) + float(row.get("showers", 0.0) or 0.0)
        if prev_r > 0.5 and curr_r < 0.2 and float(row.get("cloud_cover", 0.0) or 0.0) > 15:
            score += 12

    return float(clamp(score, 0, 100))


def score_fog(row: pd.Series) -> float:
    vis_m = float(row.get("visibility", 99999.0) or 99999.0)
    rh = float(row.get("relative_humidity_2m", 0.0) or 0.0)
    t = row.get("temperature_2m", math.nan)
    dp = row.get("dew_point_2m", math.nan)
    wind = float(row.get("wind_speed_10m", 0.0) or 0.0)

    spread = None
    if pd.notna(t) and pd.notna(dp):
        spread = abs(float(t) - float(dp))

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
    return float(clamp(score, 0, 100))


def score_storm(row: pd.Series) -> float:
    cape = float(row.get("cape", 0.0) or 0.0)
    pprob = float(row.get("precipitation_probability", 0.0) or 0.0)
    gust = float(row.get("wind_gusts_10m", 0.0) or 0.0)
    wcode = int(row.get("weather_code", 0) or 0)
    precip = float(row.get("precipitation", 0.0) or 0.0)
    pressure = float(row.get("surface_pressure", 0.0) or 0.0)

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

    return float(clamp(score, 0, 100))


# -----------------------------
# Plain-English "Good vs Watch-outs"
# -----------------------------
def explain_colour(row: pd.Series, prev_row: Optional[pd.Series]) -> Tuple[List[str], List[str], str]:
    good: List[str] = []
    bad: List[str] = []

    total = float(row.get("cloud_cover", math.nan))
    low = float(row.get("cloud_cover_low", math.nan))
    mid = float(row.get("cloud_cover_mid", math.nan))
    high = float(row.get("cloud_cover_high", math.nan))
    vis_km = float(row.get("visibility", 0.0) or 0.0) / 1000.0
    pprob = float(row.get("precipitation_probability", 0.0) or 0.0)
    gust = float(row.get("wind_gusts_10m", 0.0) or 0.0)

    if not math.isnan(total):
        if 25 <= total <= 70:
            good.append(f"Cloud amount {total:.0f}% (good for colour)")
        elif total < 25:
            bad.append(f"Mostly clear {total:.0f}% (can be plain)")
        else:
            bad.append(f"Very cloudy {total:.0f}% (may block light)")

    if not math.isnan(low):
        if low <= 25:
            good.append(f"Low cloud {low:.0f}% (horizon likely visible)")
        elif low >= 60:
            bad.append(f"Low cloud {low:.0f}% (often kills horizon glow)")
        else:
            bad.append(f"Some low cloud {low:.0f}% (watch the horizon)")

    if not math.isnan(mid) and mid >= 20:
        good.append(f"Mid cloud {mid:.0f}% (catches colour)")
    if not math.isnan(high) and high >= 20:
        good.append(f"High cloud {high:.0f}% (catches colour)")

    if vis_km >= 12:
        good.append(f"Visibility ~{vis_km:.0f} km (clean air)")
    elif vis_km <= 7 and vis_km > 0:
        bad.append(f"Visibility ~{vis_km:.0f} km (haze can mute colour)")

    if pprob >= 70:
        bad.append(f"Rain chance {pprob:.0f}% (may block light)")
    elif 40 <= pprob < 70:
        bad.append(f"Showers possible {pprob:.0f}% (breaks can be dramatic)")

    if gust >= 45:
        bad.append(f"Gusts {gust:.0f} km/h (tripod/shutter discipline)")

    if prev_row is not None:
        prev_r = float(prev_row.get("rain", 0.0) or 0.0) + float(prev_row.get("showers", 0.0) or 0.0)
        curr_r = float(row.get("rain", 0.0) or 0.0) + float(row.get("showers", 0.0) or 0.0)
        if prev_r > 0.5 and curr_r < 0.2:
            good.append("Clearing after rain (often boosts colour/contrast)")

    why = "; ".join((good + bad)[:5])
    return good[:5], bad[:5], f"Why: {why}" if why else "Why: (no strong signals)"


def explain_fog(row: pd.Series) -> Tuple[List[str], List[str], str]:
    good: List[str] = []
    bad: List[str] = []

    vis_m = float(row.get("visibility", 99999.0) or 99999.0)
    rh = float(row.get("relative_humidity_2m", math.nan))
    t = row.get("temperature_2m", math.nan)
    dp = row.get("dew_point_2m", math.nan)
    wind = float(row.get("wind_speed_10m", math.nan))

    if vis_m <= 1000:
        good.append(f"Reduced visibility ({vis_m/1000:.1f} km) (mist/fog likely)")
    elif vis_m <= 3000:
        good.append(f"Some haze ({vis_m/1000:.1f} km) (light mist possible)")
    else:
        bad.append(f"Visibility {vis_m/1000:.1f} km (fog less likely)")

    if not math.isnan(rh):
        if rh >= 95:
            good.append(f"RH {rh:.0f}% (air near saturated)")
        elif rh >= 90:
            good.append(f"RH {rh:.0f}% (humid)")
        else:
            bad.append(f"RH {rh:.0f}% (not very humid)")

    if pd.notna(t) and pd.notna(dp):
        spread = abs(float(t) - float(dp))
        if spread <= 1.5:
            good.append(f"Temp close to dew point (spread {spread:.1f}°C)")
        else:
            bad.append(f"Temp–dew spread {spread:.1f}°C (fog less likely)")

    if not math.isnan(wind):
        if wind <= 10:
            good.append(f"Light wind {wind:.0f} km/h (fog can hang around)")
        else:
            bad.append(f"Wind {wind:.0f} km/h (may disperse fog)")

    why = "; ".join((good + bad)[:5])
    return good[:5], bad[:5], f"Why: {why}" if why else "Why: (no strong signals)"


def explain_storm(row: pd.Series) -> Tuple[List[str], List[str], str]:
    good: List[str] = []
    bad: List[str] = []

    cape = float(row.get("cape", math.nan))
    pprob = float(row.get("precipitation_probability", math.nan))
    gust = float(row.get("wind_gusts_10m", math.nan))
    wcode = int(row.get("weather_code", 0) or 0)

    if not math.isnan(cape):
        if cape >= 1000:
            good.append(f"CAPE {cape:.0f} (strong storm energy)")
        elif cape >= 500:
            good.append(f"CAPE {cape:.0f} (some storm energy)")
        else:
            bad.append(f"CAPE {cape:.0f} (limited storm energy)")

    if not math.isnan(pprob):
        if pprob >= 70:
            good.append(f"Rain chance {pprob:.0f}%")
        else:
            bad.append(f"Rain chance {pprob:.0f}%")

    if not math.isnan(gust):
        if gust >= 40:
            good.append(f"Gusts {gust:.0f} km/h (drama/structure)")
        elif gust >= 25:
            good.append(f"Gusts {gust:.0f} km/h (movement/drama possible)")
        else:
            bad.append(f"Gusts {gust:.0f} km/h (less dramatic)")

    if wcode in (95, 96, 99):
        good.append("Thunderstorm indicated")
    elif wcode in (80, 81, 82):
        good.append("Showers indicated")

    why = "; ".join((good + bad)[:5])
    return good[:5], bad[:5], f"Why: {why}" if why else "Why: (no strong signals)"


# -----------------------------
# Tide events from sea_level_height_msl
# -----------------------------
def derive_tide_events(marine_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DF indexed by time with columns: kind (HIGH/LOW), height_m.
    Derived by detecting local maxima/minima in sea_level_height_msl.
    """
    if marine_df.empty or "sea_level_height_msl" not in marine_df.columns:
        return pd.DataFrame(columns=["kind", "height_m"])

    s = marine_df["sea_level_height_msl"].astype(float)
    times = s.index
    vals = s.values

    events = []
    for i in range(1, len(vals) - 1):
        a, b, c = vals[i - 1], vals[i], vals[i + 1]
        if b > a and b > c:
            events.append((times[i], "HIGH", float(b)))
        elif b < a and b < c:
            events.append((times[i], "LOW", float(b)))

    ev = pd.DataFrame(events, columns=["time", "kind", "height_m"]).set_index("time").sort_index()

    # de-noise: keep events at least 4h apart
    if not ev.empty:
        kept = []
        last_t = None
        for t, row in ev.iterrows():
            if last_t is None or (t - last_t) >= pd.Timedelta(hours=4):
                kept.append((t, row["kind"], row["height_m"]))
                last_t = t
        ev = pd.DataFrame(kept, columns=["time", "kind", "height_m"]).set_index("time").sort_index()

    return ev


def nearest_tide_event(
    tides: pd.DataFrame, t: pd.Timestamp, within_hours: float = 2.5
) -> Optional[Tuple[pd.Timestamp, str, float]]:
    # Fixed implementation (previous version could return a Timedelta by mistake)
    if tides.empty:
        return None
    t = pd.Timestamp(t)
    diffs = pd.Series(tides.index - t, index=tides.index).abs()
    nearest_time = diffs.idxmin()
    dt_hours = diffs.loc[nearest_time].total_seconds() / 3600.0
    if dt_hours <= within_hours:
        r = tides.loc[nearest_time]
        return (nearest_time, str(r["kind"]), float(r["height_m"]))
    return None


# -----------------------------
# Build opportunities
# -----------------------------
def pick_best_in_window(
    weather_df: pd.DataFrame, center: pd.Timestamp, minutes: int = 75
) -> Optional[Tuple[pd.Timestamp, float, pd.Series, Optional[pd.Series]]]:
    if weather_df.empty:
        return None

    center = pd.Timestamp(center)
    start = center - pd.Timedelta(minutes=minutes)
    end = center + pd.Timedelta(minutes=minutes)
    win = weather_df.loc[(weather_df.index >= start) & (weather_df.index <= end)]
    if win.empty:
        return None

    best: Optional[Tuple[pd.Timestamp, float, pd.Series, Optional[pd.Series]]] = None
    for t, row in win.iterrows():
        loc = weather_df.index.get_indexer([t])[0]
        prev_row = weather_df.iloc[loc - 1] if loc - 1 >= 0 else None
        s = score_colour_window(row, prev_row)
        if best is None or s > best[1]:
            best = (t, s, row, prev_row)
    return best


def build_opportunities_df(
    city: City, weather_df: pd.DataFrame, sun_df: pd.DataFrame, tides_df: pd.DataFrame
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # Sunrise/Sunset: best hour near each event, per day
    if not sun_df.empty and not weather_df.empty:
        for _, r in sun_df.iterrows():
            sr = pd.Timestamp(r["sunrise"])
            ss = pd.Timestamp(r["sunset"])

            pick_sr = pick_best_in_window(weather_df, sr, minutes=75)
            if pick_sr:
                t, score, row, prev_row = pick_sr
                tide = nearest_tide_event(tides_df, t) if city.coastal else None
                tide_txt = f"{tide[1]} {tide[0].strftime('%H:%M')} ({tide[2]:.2f}m)" if tide else ""
                good, bad, why = explain_colour(row, prev_row)
                rows.append(
                    dict(
                        type="SUNRISE",
                        time=t,
                        score=round(score, 1),
                        headline="Sunrise colour",
                        tide=tide_txt,
                        good=good,
                        watchouts=bad,
                        why=why,
                    )
                )

            pick_ss = pick_best_in_window(weather_df, ss, minutes=75)
            if pick_ss:
                t, score, row, prev_row = pick_ss
                tide = nearest_tide_event(tides_df, t) if city.coastal else None
                tide_txt = f"{tide[1]} {tide[0].strftime('%H:%M')} ({tide[2]:.2f}m)" if tide else ""
                good, bad, why = explain_colour(row, prev_row)
                rows.append(
                    dict(
                        type="SUNSET",
                        time=t,
                        score=round(score, 1),
                        headline="Sunset colour",
                        tide=tide_txt,
                        good=good,
                        watchouts=bad,
                        why=why,
                    )
                )

    # Fog: morning/evening hours only
    if not weather_df.empty:
        for t, row in weather_df.iterrows():
            hr = t.hour
            if (2 <= hr <= 10) or (18 <= hr <= 23):
                s = score_fog(row)
                if s >= 55:
                    tide = nearest_tide_event(tides_df, t) if city.coastal else None
                    tide_txt = f"{tide[1]} {tide[0].strftime('%H:%M')} ({tide[2]:.2f}m)" if tide else ""
                    good, bad, why = explain_fog(row)
                    rows.append(
                        dict(
                            type="FOG",
                            time=t,
                            score=round(s, 1),
                            headline="Fog / mist",
                            tide=tide_txt,
                            good=good,
                            watchouts=bad,
                            why=why,
                        )
                    )

    # Storm/drama: any time
    if not weather_df.empty:
        for t, row in weather_df.iterrows():
            s = score_storm(row)
            if s >= 60:
                tide = nearest_tide_event(tides_df, t) if city.coastal else None
                tide_txt = f"{tide[1]} {tide[0].strftime('%H:%M')} ({tide[2]:.2f}m)" if tide else ""
                good, bad, why = explain_storm(row)
                rows.append(
                    dict(
                        type="STORM",
                        time=t,
                        score=round(s, 1),
                        headline="Storm / drama",
                        tide=tide_txt,
                        good=good,
                        watchouts=bad,
                        why=why,
                    )
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["score", "time"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out["time_str"] = pd.to_datetime(out["time"]).dt.strftime("%a %d %b %H:%M")
    out["date"] = pd.to_datetime(out["time"]).dt.date
    return out


# -----------------------------
# Plotting (focused window)
# -----------------------------
def sun_df_in_window(sun_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if sun_df.empty:
        return sun_df
    # keep days whose sunrise/sunset falls in the window (+/- a bit)
    pad = pd.Timedelta(hours=6)
    mask = (sun_df["sunrise"] >= (start - pad)) & (sun_df["sunrise"] <= (end + pad))
    return sun_df.loc[mask]


def plot_sky(df: pd.DataFrame, sun_df: pd.DataFrame, focus_time: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Sky / Colour (no data)")
        return fig

    # Make it readable: total + low + mid/high (3–4 lines max)
    if "cloud_cover" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["cloud_cover"], mode="lines", name="Cloud total (%)"))
    if "cloud_cover_low" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["cloud_cover_low"], mode="lines", name="Cloud low (%)"))
    if "cloud_cover_mid" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["cloud_cover_mid"], mode="lines", name="Cloud mid (%)"))
    if "cloud_cover_high" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["cloud_cover_high"], mode="lines", name="Cloud high (%)"))

    # Only a few sunrise/sunset markers (windowed)
    if not sun_df.empty:
        for _, r in sun_df.iterrows():
            fig.add_vline(x=r["sunrise"], line_width=1, line_dash="dot")
            fig.add_vline(x=r["sunset"], line_width=1, line_dash="dot")

    # Mark the selected opportunity time
    fig.add_vline(x=focus_time, line_width=2)

    fig.update_layout(
        title="Sky setup for colour (focused window). Dotted lines = sunrise/sunset.",
        xaxis_title="Local time",
        yaxis_title="Percent",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title="",
    )
    return fig


def plot_fog(df: pd.DataFrame, focus_time: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Fog / Mist (no data)")
        return fig

    if "visibility" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["visibility"] / 1000.0, mode="lines", name="Visibility (km)"))
    if "relative_humidity_2m" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["relative_humidity_2m"], mode="lines", name="RH (%)"))
    if "temperature_2m" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["temperature_2m"], mode="lines", name="Temp (°C)"))
    if "dew_point_2m" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["dew_point_2m"], mode="lines", name="Dew point (°C)"))

    fig.add_vline(x=focus_time, line_width=2)

    fig.update_layout(
        title="Fog drivers (focused window): visibility, humidity, temp vs dew point",
        xaxis_title="Local time",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title="",
    )
    return fig


def plot_storm(df: pd.DataFrame, focus_time: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Storm / Drama (no data)")
        return fig

    if "cape" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["cape"], mode="lines", name="CAPE (storm energy)"))
    if "precipitation_probability" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["precipitation_probability"], mode="lines", name="Rain chance (%)"))
    if "wind_gusts_10m" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["wind_gusts_10m"], mode="lines", name="Wind gusts (km/h)"))
    if "precipitation" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["precipitation"], mode="lines", name="Precip (mm/hr)"))

    fig.add_vline(x=focus_time, line_width=2)

    fig.update_layout(
        title="Storm/drama drivers (focused window): CAPE, rain chance, gusts, precip",
        xaxis_title="Local time",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title="",
    )
    return fig


def plot_tides(marine_df: pd.DataFrame, tides_df: pd.DataFrame, sun_df: pd.DataFrame, focus_time: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    if marine_df.empty or "sea_level_height_msl" not in marine_df.columns:
        fig.update_layout(title="Tides (marine data unavailable)")
        return fig

    fig.add_trace(
        go.Scatter(x=marine_df.index, y=marine_df["sea_level_height_msl"], mode="lines", name="Sea level (incl. tides)")
    )

    if not tides_df.empty:
        highs = tides_df[tides_df["kind"] == "HIGH"]
        lows = tides_df[tides_df["kind"] == "LOW"]
        if not highs.empty:
            fig.add_trace(go.Scatter(x=highs.index, y=highs["height_m"], mode="markers", name="High tide"))
        if not lows.empty:
            fig.add_trace(go.Scatter(x=lows.index, y=lows["height_m"], mode="markers", name="Low tide"))

    if not sun_df.empty:
        for _, r in sun_df.iterrows():
            fig.add_vline(x=r["sunrise"], line_width=1, line_dash="dot")
            fig.add_vline(x=r["sunset"], line_width=1, line_dash="dot")

    fig.add_vline(x=focus_time, line_width=2)

    fig.update_layout(
        title="Tide curve (focused window). Markers = derived high/low. Dotted = sunrise/sunset.",
        xaxis_title="Local time",
        yaxis_title="Meters",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title="",
    )
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Photographer Weather Dashboard (AU)", layout="wide")

st.title("Photographer Weather Dashboard (Australia)")
st.caption(
    "Ranks opportunities for colourful sunrise/sunset, fog/mist mood, and storm/drama — and overlays tides. "
    "Uses free Open-Meteo weather + marine forecasts (no API key)."
)

with st.sidebar:
    st.header("Settings")
    city_name = st.selectbox("City", [c.name for c in CITIES], index=0)  # default Gold Coast
    days = st.slider("Forecast days", 1, 10, 3)
    top_n = st.slider("Show top N opportunities", 5, 50, 15)
    graph_window_hours = st.slider("Graph window (hours)", 12, 96, 36)

    st.divider()
    st.markdown("**Glossary (plain English)**")
    st.markdown("- **RH**: Relative Humidity (0–100%). Higher = air is closer to saturated moisture.")
    st.markdown("- **Dew point**: when temp is close to dew point, fog is more likely.")
    st.markdown("- **CAPE**: “storm energy” number. Higher can mean more thunderstorm potential.")
    st.markdown("- Tides here are model-derived for planning, not navigation.")

city = get_city(city_name)

# Fetch
weather_json = fetch_json(build_weather_url(city, days))
weather_df = hourly_to_df(weather_json)
sun_df = daily_sun_df(weather_json)

marine_df = pd.DataFrame()
tides_df = pd.DataFrame(columns=["kind", "height_m"])
if city.coastal:
    try:
        marine_json = fetch_json(build_marine_url(city, days))
        marine_df = marine_to_df(marine_json)
        tides_df = derive_tide_events(marine_df)
    except Exception:
        marine_df = pd.DataFrame()
        tides_df = pd.DataFrame(columns=["kind", "height_m"])

opps_df = build_opportunities_df(city, weather_df, sun_df, tides_df)

# At-a-glance summary
if not opps_df.empty:
    best_overall = opps_df.iloc[0]

    daily = opps_df.groupby("date")["score"].max().reset_index()
    best_day = daily.sort_values("score", ascending=False).iloc[0]

    best_sunrise = opps_df[opps_df["type"] == "SUNRISE"].iloc[0] if (opps_df["type"] == "SUNRISE").any() else None
    best_sunset = opps_df[opps_df["type"] == "SUNSET"].iloc[0] if (opps_df["type"] == "SUNSET").any() else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best next shoot", f"{best_overall['type']} {best_overall['time_str']}", f"{best_overall['score']}")
    c2.metric("Best day", f"{best_day['date']}", f"{best_day['score']:.1f}")
    if best_sunrise is not None:
        c3.metric("Best sunrise", best_sunrise["time_str"], f"{best_sunrise['score']}")
    if best_sunset is not None:
        c4.metric("Best sunset", best_sunset["time_str"], f"{best_sunset['score']}")

    with st.expander("Daily at-a-glance (max score per day)", expanded=False):
        daily2 = daily.copy()
        daily2["date"] = daily2["date"].astype(str)
        st.bar_chart(daily2.set_index("date")["score"])
else:
    st.warning("No strong opportunities detected with current thresholds. Try fewer days or different city.")

# Layout
left, right = st.columns([1.05, 1.35], gap="large")

with left:
    st.subheader("Ranked opportunities")

    if opps_df.empty:
        st.stop()

    # Table: decision-only columns (no long 'why' text)
    show_df = opps_df.head(top_n)[["rank", "type", "time_str", "score", "headline", "tide"]].copy()

    chosen_rank: int = 1

    # Try click-to-select (newer Streamlit). Fallback to number input if not supported.
    try:
        event = st.dataframe(
            show_df,
            use_container_width=True,
            height=520,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "type": st.column_config.TextColumn("Type", width="small"),
                "time_str": st.column_config.TextColumn("When", width="medium"),
                "score": st.column_config.NumberColumn("Score", width="small"),
                "headline": st.column_config.TextColumn("What", width="medium"),
                "tide": st.column_config.TextColumn("Tide", width="medium"),
            },
        )
        if event.selection.rows:
            chosen_rank = int(show_df.iloc[event.selection.rows[0]]["rank"])
        else:
            chosen_rank = int(show_df.iloc[0]["rank"])
    except TypeError:
        # Older Streamlit: no row selection support
        st.dataframe(show_df, use_container_width=True, height=520, hide_index=True)
        chosen_rank = int(
            st.number_input(
                "Select rank #",
                min_value=1,
                max_value=int(opps_df["rank"].max()),
                value=1,
                step=1,
            )
        )

    chosen = opps_df[opps_df["rank"] == chosen_rank].iloc[0]
    focus_time = pd.Timestamp(chosen["time"])

    st.markdown("### Selected recommendation")
    st.markdown(f"**{chosen['type']} — {chosen['time_str']} — score {chosen['score']}**")
    if str(chosen.get("tide", "")).strip():
        st.markdown(f"**Tide:** {chosen['tide']}")
    st.markdown(f"**{chosen['why']}**")

    gcol, bcol = st.columns(2)
    with gcol:
        st.markdown("**Good signs**")
        for x in (chosen["good"] or []):
            st.write(f"✅ {x}")
    with bcol:
        st.markdown("**Watch-outs**")
        for x in (chosen["watchouts"] or []):
            st.write(f"⚠️ {x}")

    st.markdown("**Shoot note (practical):**")
    if chosen["type"] in ("SUNRISE", "SUNSET"):
        st.write("- Arrive 30–45 min early. If low cloud thickens on the horizon, pivot to reflections/silhouettes.")
        st.write("- If it’s clearing after rain, hunt wet surfaces (sand/roads/rocks) for reflections and contrast.")
    elif chosen["type"] == "FOG":
        st.write("- Fog lies: protect highlights, lift midtones later. Look for layers (poles, headlands, bridges).")
        st.write("- Use longer focal lengths for compressed layers; keep shutter up if handheld.")
    else:
        st.write("- Safety first. Don’t chase lightning in exposed areas.")
        st.write("- Gusts can make structure epic but demand tripod discipline or deliberate motion blur.")

with right:
    st.subheader("Supporting graphs (focused on the selected time)")

    half = pd.Timedelta(hours=graph_window_hours / 2)
    start = focus_time - half
    end = focus_time + half

    weather_focus = weather_df.loc[(weather_df.index >= start) & (weather_df.index <= end)]
    marine_focus = marine_df.loc[(marine_df.index >= start) & (marine_df.index <= end)] if not marine_df.empty else marine_df
    sun_focus = sun_df_in_window(sun_df, start, end)
    tides_focus = tides_df.loc[(tides_df.index >= start) & (tides_df.index <= end)] if not tides_df.empty else tides_df

    tabs = st.tabs(["Sky / Colour", "Fog / Mist", "Storm / Drama", "Tides"])

    with tabs[0]:
        st.plotly_chart(plot_sky(weather_focus, sun_focus, focus_time), use_container_width=True)

    with tabs[1]:
        st.plotly_chart(plot_fog(weather_focus, focus_time), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(plot_storm(weather_focus, focus_time), use_container_width=True)

    with tabs[3]:
        if city.coastal:
            st.plotly_chart(plot_tides(marine_focus, tides_focus, sun_focus, focus_time), use_container_width=True)
        else:
            st.info("Selected city is inland — tide chart disabled.")

st.divider()
st.caption(
    "Reality check: this is a heuristic ranking tool. Use it to decide when to be out, then confirm with your eyes on location."
)


st.markdown(
    """
    <style>
      /* Make st.metric big-value text smaller */
      [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;   /* was ~2.5–3rem */
        line-height: 1.15 !important;
      }
      /* Optional: also tighten the label a touch */
      [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)
