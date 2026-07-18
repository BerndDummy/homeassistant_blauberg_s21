"""The Blauberg S21 integration."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from pybls21.client import S21Client
from .const import DOMAIN

# MaNi additions - additional functions via buttons, switches and select
PLATFORMS: list[Platform] = [
    Platform.CLIMATE,
    Platform.BUTTON,
    Platform.BINARY_SENSOR,
    Platform.SENSOR,
    Platform.SWITCH,
    Platform.SELECT,
]
# EO MaNi additions - additional functions via buttons, switches and select


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Blauberg S21 from a config entry."""

    hass.data.setdefault(DOMAIN, {})

    host = entry.data[CONF_HOST]
    port = entry.data[CONF_PORT]

    try:
        client = S21Client(host, port)
        await client.poll()
    except Exception as ex:
        raise ConfigEntryNotReady(
            f"Failed to connect to modbusTCP://{host}:{port}"
        ) from ex

    hass.data[DOMAIN][entry.entry_id] = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok
