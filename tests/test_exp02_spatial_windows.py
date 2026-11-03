"""The window plotter must put cells where they actually are.

A wrong PRIO-GRID offset does not raise. It produces a clean, plausible, well-scaled map of the
wrong part of the planet, and nothing in the figure says so — the failure is invisible to exactly
the eyeball check the tool exists to serve. These tests pin the geometry against points whose
answer is known independently of the code.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.exp02.plot_spatial_windows import (
    PG_SIDE,
    ZONES,
    Window,
    _top_share,
    raster,
)


def gid_of(lat: float, lon: float) -> int:
    """The PRIO-GRID id containing a point, derived here independently of the module."""
    return int((lat + 90) // 0.5) * PG_SIDE + int((lon + 180) // 0.5) + 1


@pytest.mark.green
@pytest.mark.parametrize(
    "place,lat,lon,zone",
    [
        ("Baghdad", 33.5, 44.4, "levant"),
        ("Kano", 12.0, 8.5, "nigeria"),
        ("Donetsk", 48.5, 37.8, "ukraine"),
        ("Kabul", 34.5, 69.2, "afpak"),
        ("Mogadishu", 2.0, 45.3, "horn"),
        ("Goma", -1.7, 29.2, "great_lakes"),
        ("Gao", 16.3, -0.04, "sahel"),
        ("Sittwe", 20.1, 92.9, "myanmar"),
    ],
)
def test_each_zone_contains_the_city_it_is_named_for(place, lat, lon, zone):
    """Every declared zone is checked against a point whose location is known independently.

    ⛔ Written this way after a first version parametrized four cities against ONE window and
    reported `5 passed, 3 skipped` — three quarters of the cities were never checked, and the
    green came from the skips. A roster narrower than the risk (`feedback_scope_the_check...`):
    the risk is that ANY zone's box is wrong, so every zone needs its own point.
    """
    win = Window(zone, *ZONES[zone])
    assert win.lat_min <= lat <= win.lat_max and win.lon_min <= lon <= win.lon_max, (
        f"{place} ({lat}, {lon}) is not inside the '{zone}' box "
        f"lat {win.lat_min}..{win.lat_max} lon {win.lon_min}..{win.lon_max} — the zone is "
        f"named for a region it does not cover, so its figures are mislabelled."
    )
    grid = raster(np.array([gid_of(lat, lon)]), np.array([7.0]), win)
    assert np.nansum(grid) == pytest.approx(7.0), (
        f"{place} is inside the '{zone}' box but its value did not land on the raster — "
        f"the gid->row/col conversion and the window bounds disagree."
    )


@pytest.mark.red
def test_a_cell_outside_the_window_is_not_drawn_inside_it():
    """Out-of-window cells must be dropped, not wrapped onto the edge.

    Wrapping is the failure mode of a bare modulo: a cell east of the box reappears on its west
    edge as a bright pixel in a place that has no such prediction.
    """
    win = Window("levant", 29.0, 38.0, 35.0, 49.0)
    far = gid_of(12.0, 8.5)  # Kano — same latitude band, far to the west
    grid = raster(np.array([far]), np.array([999.0]), win)
    assert np.isnan(grid).all(), (
        "a cell outside the window was drawn inside it; 999 appears somewhere on a raster "
        "whose box excludes that longitude."
    )


@pytest.mark.green
def test_the_raster_is_nan_where_there_is_no_land_not_zero():
    """Absent cell and zero-fatality cell are different claims and must render differently."""
    win = Window("tiny", 0.0, 1.0, 0.0, 1.0)
    grid = raster(np.array([gid_of(0.25, 0.25)]), np.array([0.0]), win)
    assert (grid == 0.0).sum() == 1, "the one present cell should hold its zero value"
    assert np.isnan(grid).sum() == grid.size - 1, (
        "cells with no land must be NaN; filling them with 0 invents land and erases the "
        "coastline that lets a reader confirm the window is the place it is labelled."
    )


@pytest.mark.green
def test_concentration_rises_when_the_same_mass_is_packed_into_fewer_cells():
    """`_top_share` must actually measure clustering, in the direction it claims."""
    spread = np.full(1000, 1.0)
    packed = np.zeros(1000)
    packed[:10] = 100.0
    assert _top_share(packed, 0.01) > _top_share(spread, 0.01), (
        "a perfectly concentrated field did not score above a uniform one of the same total "
        "mass, so the statistic printed beside each panel does not measure what it says."
    )
    assert _top_share(packed, 0.01) == pytest.approx(1.0)


@pytest.mark.red
def test_concentration_refuses_rather_than_dividing_by_an_empty_field():
    """An all-zero window must not report a concentration of 0/0 as if it were a number."""
    assert np.isnan(_top_share(np.zeros(500), 0.01)), (
        "an all-zero field produced a finite concentration; that is a number with no content "
        "printed in a title a human is being asked to compare against another panel."
    )
