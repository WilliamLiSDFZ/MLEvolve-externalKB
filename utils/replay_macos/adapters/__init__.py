"""Select a reviewed parent port by its original source SHA256."""

from types import ModuleType

from . import parent_36845c7c, parent_f327226c


_ADAPTERS = {
    parent_36845c7c.ORIGINAL_SOURCE_SHA256: parent_36845c7c,
    parent_f327226c.ORIGINAL_SOURCE_SHA256: parent_f327226c,
}


def get_adapter(source_sha256: str) -> ModuleType:
    """Return a module exposing ADAPTER_VERSION and adapt(source)."""
    try:
        return _ADAPTERS[source_sha256]
    except KeyError:
        raise ValueError(f"Unsupported replay parent SHA256: {source_sha256}") from None
