"""Compatibility facade: one schema and connection implementation."""
from observability.debug_store import get_database_url, get_engine, get_connection, init_db

__all__ = ["get_database_url", "get_engine", "get_connection", "init_db"]
