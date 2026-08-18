"""Reach Postgres through the agent proxy's CONNECT tunnel.

Sandboxed sessions have no direct outbound TCP: everything must go
through the policy proxy, which speaks HTTP CONNECT.  libpq has no idea
what an HTTP proxy is, so ``psycopg2.connect()`` to a cloud database
simply hangs — even when the policy *allows* the destination.

This bridges the two.  A local listener accepts libpq's ordinary TCP
connection, opens its own connection to the proxy, issues

    CONNECT db-host:5432 HTTP/1.1

and then pipes bytes in both directions.  Nothing bypasses the policy —
the tunnel exists only if the gateway answers 200 to that CONNECT, and a
denied host fails exactly as it should.

TLS is untouched: libpq negotiates directly with the database through
the tunnel.  For that to work the connection must present the *real*
hostname for SNI and certificate checks while dialing the local port,
which is what libpq's ``hostaddr`` is for — ``host`` names the server,
``hostaddr`` says where to send packets.  ``connect_via_proxy`` wires
that up.

    from big2.pgtunnel import connect_via_proxy
    with connect_via_proxy(os.environ["DATABASE_URL"]) as con:
        ...
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import urllib.parse
from typing import Iterator, Optional, Tuple

PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")


def proxy_address() -> Optional[Tuple[str, int]]:
    for key in PROXY_ENV:
        raw = os.environ.get(key)
        if raw:
            u = urllib.parse.urlparse(raw if "://" in raw else f"http://{raw}")
            if u.hostname and u.port:
                return u.hostname, u.port
    return None


def _open_tunnel(proxy: Tuple[str, int], host: str, port: int,
                 timeout: float = 20.0) -> socket.socket:
    """One CONNECT-tunnelled socket to host:port, or an error."""
    s = socket.create_connection(proxy, timeout=timeout)
    s.sendall(
        f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode()
    )
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = s.recv(1024)
        if not chunk:
            s.close()
            raise ConnectionError("proxy closed during CONNECT")
        head += chunk
        if len(head) > 65536:
            break
    status = head.split(b"\r\n", 1)[0].decode(errors="replace")
    if " 200" not in status:
        s.close()
        raise ConnectionError(f"proxy refused the tunnel: {status.strip()}")
    s.settimeout(None)
    return s


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)


class ProxyTunnel:
    """A local port that forwards to ``host:port`` through the proxy."""

    def __init__(self, host: str, port: int = 5432,
                 proxy: Optional[Tuple[str, int]] = None):
        self.host = host
        self.port = port
        self.proxy = proxy or proxy_address()
        if not self.proxy:
            raise RuntimeError("no HTTPS_PROXY configured for this session")
        self._srv: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.local_port = 0

    def start(self) -> "ProxyTunnel":
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self.local_port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def _serve(self) -> None:
        assert self._srv is not None
        while not self._stop.is_set():
            try:
                client, _ = self._srv.accept()
            except OSError:
                break
            try:
                upstream = _open_tunnel(self.proxy, self.host, self.port)
            except Exception:
                with contextlib.suppress(OSError):
                    client.close()
                continue
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=_pump, args=(a, b),
                                 daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        if self._srv is not None:
            with contextlib.suppress(OSError):
                self._srv.close()

    def __enter__(self) -> "ProxyTunnel":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()


def tunnel_dsn(url: str) -> Tuple[str, ProxyTunnel]:
    """(libpq DSN pointed at a live local tunnel, the tunnel).

    Keeps the original hostname in ``host`` so SNI and certificate
    verification still target the real server, and sends the packets to
    the tunnel via ``hostaddr``.
    """
    u = urllib.parse.urlparse(url)
    if not u.hostname:
        raise ValueError("connection string has no host")
    tun = ProxyTunnel(u.hostname, u.port or 5432).start()
    params = dict(urllib.parse.parse_qsl(u.query))
    # The tunnel is a plain byte pipe, so TLS is still end-to-end; channel
    # binding, however, depends on the exact TLS endpoint and is dropped
    # rather than risking a SCRAM mismatch.
    params.pop("channel_binding", None)
    params.setdefault("sslmode", "require")
    parts = [
        f"host={u.hostname}",
        "hostaddr=127.0.0.1",
        f"port={tun.local_port}",
        f"dbname={(u.path or '/').lstrip('/')}",
    ]
    if u.username:
        parts.append(f"user={u.username}")
    if u.password:
        parts.append(f"password={urllib.parse.unquote(u.password)}")
    parts += [f"{k}={v}" for k, v in params.items()]
    return " ".join(parts), tun


@contextlib.contextmanager
def connect_via_proxy(url: str, **kwargs) -> Iterator["object"]:
    """psycopg2 connection to a database only reachable via the proxy."""
    import psycopg2

    dsn, tun = tunnel_dsn(url)
    con = None
    try:
        con = psycopg2.connect(dsn, **kwargs)
        yield con
    finally:
        if con is not None:
            with contextlib.suppress(Exception):
                con.close()
        tun.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("DATABASE_URL"),
                        help="postgres:// connection string")
    parser.add_argument("--query", default="SELECT 1")
    args = parser.parse_args()
    if not args.url:
        raise SystemExit("no --url and no DATABASE_URL")
    with connect_via_proxy(args.url) as con:
        cur = con.cursor()
        cur.execute(args.query)
        for row in cur.fetchall():
            print(row)


if __name__ == "__main__":
    main()
