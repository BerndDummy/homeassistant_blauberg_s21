"""Additional Blauberg S21 water-heater monitoring sensors."""
from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    UpdateFailed,
)

from pybls21.client import S21Client

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# S21 input registers from the official Modbus table.
IR_RETURN_WATER_TEMPERATURE = 8
IR_MAIN_HEATER_OUTPUT = 44
IR_RETURN_WATER_OUTPUT = 47
IR_EXTENDED_START = IR_RETURN_WATER_TEMPERATURE
IR_EXTENDED_COUNT = IR_RETURN_WATER_OUTPUT - IR_EXTENDED_START + 1

# Discrete inputs 7..18 include heater state and return-water preheating state.
DI_MONITORING_START = 7
DI_MONITORING_COUNT = 12

# Conservative monitoring thresholds. These do not alter Blauberg control.
FROST_WARNING_TEMPERATURE = 10.0
FROST_CRITICAL_TEMPERATURE = 5.0


def _signed_16bit(value: int) -> int:
    """Convert an unsigned Modbus register to signed 16-bit."""
    return value - 0x10000 if value > 0x7FFF else value


def _decode_temperature(value: int) -> float | None:
    """Decode an S21 temperature register in tenths of a degree Celsius."""
    signed = _signed_16bit(value)
    if signed in (-32768, 32767):
        return None
    return signed / 10


def _clamp_percent(value: int) -> int:
    """Return the physical 0-100 percent part of a PID output."""
    return max(0, min(100, int(value)))


class BlS21ExtendedRegisterCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Read water-heater registers not exposed by pybls21 4.6.1."""

    def __init__(self, hass: HomeAssistant, client: S21Client) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name="Blauberg S21 water-heater monitoring",
            update_interval=timedelta(seconds=30),
        )
        self.client = client

    async def _async_update_data(self) -> dict[str, Any]:
        async def _read_monitoring_data() -> tuple[list[int], list[bool]]:
            registers = await self.client._read_input_registers(
                IR_EXTENDED_START,
                count=IR_EXTENDED_COUNT,
            )
            discrete_inputs = await self.client._read_discrete_inputs(
                DI_MONITORING_START,
                count=DI_MONITORING_COUNT,
            )
            return registers, discrete_inputs

        try:
            registers, discrete_inputs = await self.client._do_with_connection(
                _read_monitoring_data
            )
        except Exception as err:
            raise UpdateFailed(
                f"Unable to read Blauberg water-heater registers: {err}"
            ) from err

        return_water_temperature = _decode_temperature(
            registers[IR_RETURN_WATER_TEMPERATURE - IR_EXTENDED_START]
        )
        main_heater_raw = int(registers[IR_MAIN_HEATER_OUTPUT - IR_EXTENDED_START])
        return_water_raw = int(registers[IR_RETURN_WATER_OUTPUT - IR_EXTENDED_START])

        main_heater_percent = _clamp_percent(main_heater_raw)
        return_water_percent = _clamp_percent(return_water_raw)

        return {
            "return_water_temperature": return_water_temperature,
            "main_heater_output": main_heater_percent,
            "main_heater_output_raw": main_heater_raw,
            "return_water_output": return_water_percent,
            "return_water_output_raw": return_water_raw,
            # The physical water valve must satisfy both normal reheating and
            # return-water protection. The larger controller demand is therefore
            # the useful combined valve demand for monitoring.
            "water_heater_demand": max(main_heater_percent, return_water_percent),
            "fan_assist_active": main_heater_raw > 100,
            "heater_active": bool(discrete_inputs[7 - DI_MONITORING_START]),
            "main_heater_thermostat": bool(
                discrete_inputs[11 - DI_MONITORING_START]
            ),
            "water_pressure_ok": bool(discrete_inputs[14 - DI_MONITORING_START]),
            "water_flow_ok": bool(discrete_inputs[15 - DI_MONITORING_START]),
            "water_preheating_active": bool(
                discrete_inputs[18 - DI_MONITORING_START]
            ),
        }


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the additional Blauberg S21 sensors."""
    client: S21Client = hass.data[DOMAIN][config_entry.entry_id]
    coordinator = BlS21ExtendedRegisterCoordinator(hass, client)
    await coordinator.async_config_entry_first_refresh()

    async_add_entities(
        [
            BlS21ReturnWaterTemperatureSensor(coordinator, client, config_entry),
            BlS21WaterHeaterDemandSensor(coordinator, client, config_entry),
            BlS21FrostMonitoringSensor(coordinator, client, config_entry),
        ]
    )


class BlS21MonitoringSensor(
    CoordinatorEntity[BlS21ExtendedRegisterCoordinator], SensorEntity
):
    """Base class for water-heater monitoring sensors."""

    _attr_has_entity_name = False

    def __init__(
        self,
        coordinator: BlS21ExtendedRegisterCoordinator,
        client: S21Client,
        config_entry: ConfigEntry,
        unique_suffix: str,
    ) -> None:
        super().__init__(coordinator)
        self._client = client
        self._config_entry = config_entry
        base_unique_id = (
            config_entry.unique_id
            or (client.device.unique_id if client.device else config_entry.entry_id)
        )
        self._attr_unique_id = f"{base_unique_id}_{unique_suffix}"

    @property
    def device_info(self) -> DeviceInfo:
        """Attach the sensors to the existing Blauberg device."""
        base_unique_id = (
            self._config_entry.unique_id
            or (
                self._client.device.unique_id
                if self._client.device
                else self._config_entry.entry_id
            )
        )
        return DeviceInfo(
            identifiers={(DOMAIN, base_unique_id)},
            name=(
                self._client.device.name
                if self._client.device and self._client.device.name
                else self._config_entry.title
            ),
            manufacturer=(
                self._client.device.manufacturer if self._client.device else "Blauberg"
            ),
            model=self._client.device.model if self._client.device else "S21",
            sw_version=self._client.device.sw_version if self._client.device else None,
        )


class BlS21ReturnWaterTemperatureSensor(BlS21MonitoringSensor):
    """Return-water temperature sensor T5."""

    _attr_name = "Blauberg Nachheizregister Rücklauf"
    _attr_icon = "mdi:thermometer-water"
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_suggested_display_precision = 1

    def __init__(
        self,
        coordinator: BlS21ExtendedRegisterCoordinator,
        client: S21Client,
        config_entry: ConfigEntry,
    ) -> None:
        super().__init__(coordinator, client, config_entry, "return_water_temperature")

    @property
    def native_value(self) -> float | None:
        return self.coordinator.data.get("return_water_temperature")


class BlS21WaterHeaterDemandSensor(BlS21MonitoringSensor):
    """Effective water-heater valve demand."""

    _attr_name = "Blauberg Nachheizregister Ventilanforderung"
    _attr_icon = "mdi:valve"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_suggested_display_precision = 0

    def __init__(
        self,
        coordinator: BlS21ExtendedRegisterCoordinator,
        client: S21Client,
        config_entry: ConfigEntry,
    ) -> None:
        super().__init__(coordinator, client, config_entry, "water_heater_demand")

    @property
    def native_value(self) -> int | None:
        return self.coordinator.data.get("water_heater_demand")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "temperature_control_output": self.coordinator.data.get(
                "main_heater_output"
            ),
            "return_water_protection_output": self.coordinator.data.get(
                "return_water_output"
            ),
            "temperature_control_output_raw": self.coordinator.data.get(
                "main_heater_output_raw"
            ),
            "return_water_protection_output_raw": self.coordinator.data.get(
                "return_water_output_raw"
            ),
            "fan_assist_active": self.coordinator.data.get("fan_assist_active"),
            "heater_active": self.coordinator.data.get("heater_active"),
            "water_preheating_active": self.coordinator.data.get(
                "water_preheating_active"
            ),
        }


class BlS21FrostMonitoringSensor(BlS21MonitoringSensor):
    """Monitoring state for the water reheater frost risk."""

    _attr_name = "Blauberg Nachheizregister Frostüberwachung"

    def __init__(
        self,
        coordinator: BlS21ExtendedRegisterCoordinator,
        client: S21Client,
        config_entry: ConfigEntry,
    ) -> None:
        super().__init__(coordinator, client, config_entry, "frost_monitoring")

    @property
    def native_value(self) -> str:
        temperature = self.coordinator.data.get("return_water_temperature")
        protection_output = self.coordinator.data.get("return_water_output", 0)
        preheating_active = self.coordinator.data.get(
            "water_preheating_active", False
        )

        if temperature is None:
            return "Sensor nicht verfügbar"
        if temperature <= FROST_CRITICAL_TEMPERATURE:
            return "Frostgefahr"
        if preheating_active or protection_output > 0:
            return "Frostschutz regelt"
        if temperature < FROST_WARNING_TEMPERATURE:
            return "Beobachten"
        return "Normal"

    @property
    def icon(self) -> str:
        state = self.native_value
        if state == "Normal":
            return "mdi:snowflake-check"
        if state == "Frostschutz regelt":
            return "mdi:snowflake-alert"
        if state == "Beobachten":
            return "mdi:thermometer-alert"
        if state == "Frostgefahr":
            return "mdi:alert-octagon"
        return "mdi:help-circle-outline"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "return_water_temperature": self.coordinator.data.get(
                "return_water_temperature"
            ),
            "water_heater_demand": self.coordinator.data.get(
                "water_heater_demand"
            ),
            "return_water_protection_output": self.coordinator.data.get(
                "return_water_output"
            ),
            "heater_active": self.coordinator.data.get("heater_active"),
            "water_preheating_active": self.coordinator.data.get(
                "water_preheating_active"
            ),
            "water_pressure_ok": self.coordinator.data.get("water_pressure_ok"),
            "water_flow_ok": self.coordinator.data.get("water_flow_ok"),
            "alarm_state": (
                self._client.device.alarm_state if self._client.device else None
            ),
            "alarm_codes": (
                self._client.device.alarm_codes if self._client.device else []
            ),
            "warning_temperature": FROST_WARNING_TEMPERATURE,
            "critical_temperature": FROST_CRITICAL_TEMPERATURE,
            "monitoring_only": True,
        }
