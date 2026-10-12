"""Defense in depth: prohibit Python network connections outside loopback."""
import ipaddress
import sys

_installed = False


def is_loopback(host):
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _network_audit(event, args):
    if event in {"socket.connect", "socket.sendto"}:
        address = args[1]
        # Local UNIX sockets are used by OS / numerical libraries.
        if isinstance(address, tuple) and not is_loopback(address[0]):
            raise PermissionError("Local-only mode blocks non-loopback network traffic.")
    elif event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        host = args[0]
        if host is not None and not is_loopback(host):
            raise PermissionError("Local-only mode blocks external DNS lookup.")


def install_network_guard():
    global _installed
    if not _installed:
        sys.addaudithook(_network_audit)
        _installed = True
