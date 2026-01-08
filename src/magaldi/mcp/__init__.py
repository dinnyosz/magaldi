"""Magaldi MCP Server module.

Provides Model Context Protocol integration for code discovery.
"""

from magaldi.mcp.server import MagaldiMCPServer, run_server


def main() -> None:
    """Entry point for MCP server."""
    run_server()


def create_server() -> MagaldiMCPServer:
    """Factory function for MCP server discovery."""
    return MagaldiMCPServer()


__all__ = ["MagaldiMCPServer", "main", "create_server", "run_server"]
