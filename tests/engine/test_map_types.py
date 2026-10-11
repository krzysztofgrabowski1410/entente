"""Tests for the core board identity types (Province, Location)."""

import dataclasses

import pytest

from entente.engine.map import Location, Province, Terrain


def test_location_str_without_coast() -> None:
    assert str(Location("PAR")) == "PAR"


def test_location_str_with_coast() -> None:
    assert str(Location("STP", "NC")) == "STP/NC"


def test_locations_with_different_coasts_are_distinct() -> None:
    assert Location("STP", "NC") != Location("STP", "SC")
    assert Location("STP", "NC") != Location("STP")


def test_location_is_hashable_value_object() -> None:
    assert {Location("BUL", "EC"), Location("BUL", "EC")} == {Location("BUL", "EC")}


def test_location_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Location("PAR").province = "BUR"  # type: ignore[misc]


def test_province_defaults_to_classic_diplomacy() -> None:
    province = Province(key="PAR", name="Paris", terrain=Terrain.INLAND)
    assert province.capacity == 1
    assert not province.is_supply_center


def test_province_capacity_none_means_unbounded() -> None:
    angband = Province(key="ANG", name="Angband", terrain=Terrain.INLAND, capacity=None)
    assert angband.capacity is None
