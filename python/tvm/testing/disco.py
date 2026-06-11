# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Testing utilities for the Disco distributed runtime."""

# Defer annotation evaluation: `tvm.runtime.disco` is None on builds without
# the disco runtime, and this module must still be importable there.
from __future__ import annotations

import socket
import subprocess
import sys
import threading

from tvm.runtime import disco as di

_SOCKET_SESSION_TESTER = None


def _get_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class SocketSessionTester:
    """Run a disco SocketSession with one local node and remote nodes.

    Each remote node is a `tvm.exec.disco_remote_socket_session` subprocess
    launched with the current Python interpreter.
    """

    def __init__(self, num_workers, num_nodes=2, num_groups=1):
        # Initialize the attributes used by __del__ first, so that teardown is
        # safe even when __init__ raises below.
        self.sess = None
        self.remote_nodes = []
        assert num_workers % num_nodes == 0
        num_workers_per_node = num_workers // num_nodes
        server_host = "localhost"
        server_port = _get_free_port()
        server_exc = []

        def start_server():
            try:
                self.sess = di.SocketSession(
                    num_nodes, num_workers_per_node, num_groups, server_host, server_port
                )
            except Exception as exc:  # pylint: disable=broad-except
                server_exc.append(exc)

        thread = threading.Thread(target=start_server)
        thread.start()

        cmd = "tvm.exec.disco_remote_socket_session"
        for _i in range(num_nodes - 1):
            self.remote_nodes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        cmd,
                        server_host,
                        str(server_port),
                        str(num_workers_per_node),
                    ],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
            )

        thread.join()
        if server_exc:
            raise server_exc[0]

    # Bound at class creation: module globals may already be cleared when
    # __del__ runs during interpreter shutdown.
    _TIMEOUT_EXPIRED = subprocess.TimeoutExpired

    def __del__(self):
        try:
            # Shut down the session first so remote nodes can exit gracefully.
            if self.sess is not None:
                self.sess.shutdown()
        finally:
            for node in self.remote_nodes:
                try:
                    node.wait(timeout=10)
                except self._TIMEOUT_EXPIRED:
                    node.kill()
                    node.wait()


def create_socket_session(num_workers) -> di.Session:
    """Create a socket session backed by one local and one remote node.

    The tester is kept alive in a module-level global so that the session
    survives until the next call (or interpreter exit) replaces it.
    """
    global _SOCKET_SESSION_TESTER
    # Rebind (not `del`) so the global stays defined if the constructor raises.
    _SOCKET_SESSION_TESTER = None
    _SOCKET_SESSION_TESTER = SocketSessionTester(num_workers)
    assert _SOCKET_SESSION_TESTER.sess is not None
    return _SOCKET_SESSION_TESTER.sess
