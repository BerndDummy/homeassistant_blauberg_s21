"""Sensor entities for the Blauberg S21 integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfPressure, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from pybls21.client import S21Client

from .const import DOMAIN


@dataclass(frozen=True, kw_only=True)
class BlaubergS21SensorEntityDescription(SensorEntityDescription):
    """Describe a Blauberg S21 sensor."""

    value_fn: Callable[[Any], Any]


SENSOR_DESCRIPTIONS: tuple[BlaubergS21SensorEntityDescription, ...] = (
    BlaubergS21SensorEntityDescription(
        key="intake_temperature_in",
        translation_key="intake_temperature_in",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:thermometer-chevron-down",
        value_fn=lambda device: device.current_intake_temperature,
    ),
    BlaubergS21SensorEntityDescription(
        key="intake_temperature_out",
        translation_key="intake_temperature_out",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:thermometer-chevron-up",
        value_fn=lambda device: device.current_intake_temperature_out,
    ),
    BlaubergS21SensorEntityDescription(
        key="outlet_temperature_in",
        translation_key="outlet_temperature_in",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:home-thermometer-outline",
        value_fn=lambda device: device.current_outlet_temperature_in,
    ),
    BlaubergS21SensorEntityDescription(
        key="outlet_temperature_out",
        translation_key="outlet_temperature_out",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:home-export-outline",
        value_fn=lambda device: device.current_outlet_temperature_out,
    ),
    BlaubergS21SensorEntityDescription(
        key="incoming_air_pressure",
        translation_key="incoming_air_pressure",
        device_class=SensorDeviceClass.PRESSURE,
        native_unit_of_measurement=UnitOfPressure.PA,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:gauge",
        value_fn=lambda device: device.pressure_air_incoming,
    ),
    BlaubergS21SensorEntityDescription(
        key="outgoing_air_pressure",
        translation_key="outgoing_air_pressure",
        device_class=SensorDeviceClass.PRESSURE,
        native_unit_of_measurement=UnitOfPressure.PA,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:gauge",
        value_fn=lambda device: device.pressure_air_outgoing,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Blauberg S21 sensors."""
    client: S21Client = hass.data[DOMAIN][config_entry.entry_id]
    async_add_entities(
        BlaubergS21Sensor(client, config_entry, description)
        for description in SENSOR_DESCRIPTIONS
    )


class BlaubergS21Sensor(SensorEntity):
    """Representation of a Blauberg S21 sensor."""

    _attr_has_entity_name = True
    entity_description: BlaubergS21SensorEntityDescription

    def __init__(
        self,
        client: S21Client,
        config_entry: ConfigEntry,
        description: BlaubergS21SensorEntityDescription,
    ) -> None:
        self._client = client
        self._device_id = config_entry.unique_id or config_entry.entry_id
        self.entity_description = description
        self._attr_unique_id = f"{self._device_id}_{description.key}"

    @property
    def available(self) -> bool:
        """Return whether the device and this value are available."""
        return bool(
            self._client.device
            and self._client.device.available
            and self.native_value is not None
        )

    @property
    def native_value(self) -> Any:
        """Return the sensor value."""
        if not self._client.device:
            return None
        return self.entity_description.value_fn(self._client.device)

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device registry information."""
        return DeviceInfo(identifiers={(DOMAIN, self._device_id)})
