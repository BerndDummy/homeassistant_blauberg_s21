"""Binary sensor entities for the Blauberg S21 integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from pybls21.client import S21Client

from .const import DOMAIN


@dataclass(frozen=True, kw_only=True)
class BlaubergS21BinarySensorEntityDescription(BinarySensorEntityDescription):
    """Describe a Blauberg S21 binary sensor."""

    value_fn: Callable[[Any], bool | None]
    attributes_fn: Callable[[Any], dict[str, Any]] | None = None


BINARY_SENSOR_DESCRIPTIONS: tuple[
    BlaubergS21BinarySensorEntityDescription, ...
] = (
    BlaubergS21BinarySensorEntityDescription(
        key="connection",
        translation_key="connection",
        device_class=BinarySensorDeviceClass.CONNECTIVITY,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda device: device.available,
    ),
    BlaubergS21BinarySensorEntityDescription(
        key="alarm",
        translation_key="alarm",
        device_class=BinarySensorDeviceClass.PROBLEM,
        value_fn=lambda device: (
            device.alarm_state != 0 if device.alarm_state is not None else None
        ),
        attributes_fn=lambda device: {"codes": device.alarm_codes},
    ),
    BlaubergS21BinarySensorEntityDescription(
        key="filter_warning",
        translation_key="filter_warning",
        device_class=BinarySensorDeviceClass.PROBLEM,
        value_fn=lambda device: (
            device.filter_state != 0 if device.filter_state is not None else None
        ),
        attributes_fn=lambda device: {"countdown": device.filter_countdown},
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Blauberg S21 binary sensors."""
    client: S21Client = hass.data[DOMAIN][config_entry.entry_id]
    async_add_entities(
        BlaubergS21BinarySensor(client, config_entry, description)
        for description in BINARY_SENSOR_DESCRIPTIONS
    )


class BlaubergS21BinarySensor(BinarySensorEntity):
    """Representation of a Blauberg S21 binary sensor."""

    _attr_has_entity_name = True
    entity_description: BlaubergS21BinarySensorEntityDescription

    def __init__(
        self,
        client: S21Client,
        config_entry: ConfigEntry,
        description: BlaubergS21BinarySensorEntityDescription,
    ) -> None:
        self._client = client
        self._device_id = config_entry.unique_id or config_entry.entry_id
        self.entity_description = description
        self._attr_unique_id = f"{self._device_id}_{description.key}"

    @property
    def available(self) -> bool:
        """Return whether the binary sensor value is available."""
        if not self._client.device:
            return False
        if self.entity_description.key == "connection":
            return True
        return bool(self._client.device.available and self.is_on is not None)

    @property
    def is_on(self) -> bool | None:
        """Return the binary sensor state."""
        if not self._client.device:
            return None
        return self.entity_description.value_fn(self._client.device)

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return additional diagnostic details."""
        if not self._client.device or not self.entity_description.attributes_fn:
            return None
        return self.entity_description.attributes_fn(self._client.device)

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device registry information."""
        return DeviceInfo(identifiers={(DOMAIN, self._device_id)})
