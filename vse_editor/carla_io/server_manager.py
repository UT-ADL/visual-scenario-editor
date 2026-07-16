"""CarlaServerManager (moved verbatim from vse.py): local CARLA server
start/attach/stop, port probing, auto-stop policy and PID hand-off via
VSE_SERVER_PID across editor self-relaunches.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

import carla

logger = logging.getLogger(__name__)

class CarlaServerManager:
    """
    Manages the lifecycle of the CARLA server process, including starting, stopping, and port management.
    Ensures the editor can connect to a running CARLA instance or launch its own.
    """

    # How long stop_server() waits for the whole server process group to exit
    # after SIGTERM before escalating to SIGKILL, and after SIGKILL before
    # giving up with a warning. Must fit inside the 15 s atexit cleanup budget.
    _stop_sigterm_wait_s = 10.0
    _stop_sigkill_wait_s = 3.0

    def __init__(self, carla_path=None, port=2000):
        # Use CARLA_ROOT environment variable if available
        if carla_path is None:
            carla_root = os.environ.get('CARLA_ROOT')
            if carla_root:
                self.carla_path = os.path.join(carla_root, 'CarlaUE4.sh')
            else:
                self.carla_path = "./CarlaUE4.sh"
        else:
            self.carla_path = carla_path
        self.port = port
        self.process = None
        self._server_log_handle = None
        self.use_existing_server = False
        self.allow_auto_stop = True
        self._atexit_registered = False
        self.known_server_pid = None
        self.server_pgid = None
        self.assume_existing_server_pid = None

        env_pid = os.environ.get('VSE_SERVER_PID')
        if env_pid:
            try:
                self.assume_existing_server_pid = int(env_pid)
            except ValueError:
                self.assume_existing_server_pid = None
            os.environ.pop('VSE_SERVER_PID', None)
        
    def check_port_available(self, port):
        """Check if a port is available"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            return result != 0  # Port is available if connection failed
        except Exception:
            # Socket error - assume port is available
            logger.debug("Socket error checking port %d, assuming available", port)
            sock.close()
            return True
    
    def find_available_port(self, start_port=2000, max_attempts=10):
        """Find an available port starting from start_port"""
        for i in range(max_attempts):
            port = start_port + i
            if self.check_port_available(port):
                return port
        raise Exception(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")

    def set_port(self, port):
        """Update the managed CARLA port."""
        self.port = port

    def _probe_server_health(self, client, port):
        """Validate an existing CARLA server is responsive before reusing it."""
        try:
            world = client.get_world()
            try:
                world.wait_for_tick(3.0)
            except Exception:
                pass
            world.get_settings()
            return True
        except Exception as exc:
            print(f"Existing CARLA server on port {port} failed health probe: {exc}")
            return False

    def _capture_server_pid(self, port):
        """Best-effort capture of the CARLA server PID for managed shutdowns."""
        pid = None
        try:
            result = subprocess.run(['lsof', '-t', f'-i:{port}'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if lines:
                    pid = int(lines[0])
        except Exception as exc:
            print(f"Warning: Unable to detect CARLA PID via lsof on port {port}: {exc}")

        if pid is None:
            try:
                result = subprocess.run(['pgrep', '-f', 'CarlaUE4'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                    if lines:
                        pid = int(lines[0])
            except Exception as exc:
                print(f"Warning: Unable to detect CARLA PID via pgrep: {exc}")

        if pid:
            self.known_server_pid = pid
            # Remember the process group now, while the process is alive --
            # at stop time the group leader (CarlaUE4.sh) may already be gone
            # and os.getpgid() would fail, orphaning the UE4 binary.
            try:
                self.server_pgid = os.getpgid(pid)
            except OSError:
                self.server_pgid = None
        return pid
    
    def check_existing_carla_server(self, port):
        """Check if there's already a CARLA server running on the port"""
        try:
            client = carla.Client('127.0.0.1', port)
            client.set_timeout(2.0)
            version = client.get_server_version()
            if not self._probe_server_health(client, port):
                return False
            print(f"Found existing CARLA server on port {port}, version: {version}")
            self._capture_server_pid(port)
            return True
        except Exception:
            # No server running or connection failed - this is expected
            return False
    
    def kill_existing_carla_processes(self):
        """Kill any existing CARLA processes"""
        try:
            # Find and kill existing CARLA processes
            result = subprocess.run(['pgrep', '-f', 'CarlaUE4'], capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                print(f"Found existing CARLA processes: {pids}")
                for pid in pids:
                    if pid.strip():
                        try:
                            print(f"Killing CARLA process {pid}")
                            os.kill(int(pid), signal.SIGTERM)
                            time.sleep(2)
                            # Force kill if still running
                            try:
                                os.kill(int(pid), signal.SIGKILL)
                            except (OSError, ProcessLookupError):
                                # Process already terminated - this is fine
                                pass
                        except Exception as e:
                            print(f"Failed to kill process {pid}: {e}")
                            
                # Wait a moment for cleanup
                time.sleep(5)  # Increased cleanup wait time
                self.known_server_pid = None
                self.server_pgid = None
                self.use_existing_server = False
                self.process = None
        except Exception as e:
            print(f"Error checking for existing CARLA processes: {e}")

    def start_server(self):
        """Start CARLA server with RenderOffScreen"""
        if self.assume_existing_server_pid:
            pid = self.assume_existing_server_pid
            print(f"Attaching to existing CARLA server (PID: {pid})")
            if not self.check_existing_carla_server(self.port):
                raise Exception("Expected CARLA server is not available on the specified port")
            self.use_existing_server = False
            self.process = None
            self.known_server_pid = pid
            self.assume_existing_server_pid = None
            return None

        # Check if CARLA is already running on the desired port
        if self.check_existing_carla_server(self.port):
            print(f"Using existing CARLA server on port {self.port}")
            self.use_existing_server = True
            pid = self._capture_server_pid(self.port)
            if pid:
                print(f"Tracking existing CARLA server PID: {pid}")
            return None
        
        # Check if port is available
        if not self.check_port_available(self.port):
            print(f"Port {self.port} is not available, killing existing CARLA processes...")
            self.kill_existing_carla_processes()
            
            # Try to find an available port
            try:
                new_port = self.find_available_port(self.port)
                print(f"Using alternative port: {new_port}")
                self.port = new_port
            except Exception as e:
                print(f"Could not find available port: {e}")
                raise
        
        if not os.path.exists(self.carla_path):
            raise FileNotFoundError(f"CARLA executable not found at: {self.carla_path}")
        
        print(f"Starting CARLA server at: {self.carla_path}")
        
        # Launch CARLA with RenderOffScreen and other optimal settings
        cmd = [
            self.carla_path,
            "-RenderOffScreen",
            "-prefernvidia",
            "-ResX=1",
            "-ResY=1"
        ]
        
        print(f"Command: {' '.join(cmd)}")

        # Capture the server's stdout/stderr to a log file so UE4 messages and any crash
        # (e.g. "Signal 11 Segmentation fault") are recoverable for debugging. Truncated each
        # launch. Override the path with VSE_CARLA_SERVER_LOG. The crashinfo dir under
        # ~/.config/Epic/CarlaUE4/Saved/Crashes/ is named with the same PID printed below, so the
        # log, the crash dump, and the VSE-side log can all be correlated by that PID.
        server_log_path = os.environ.get("VSE_CARLA_SERVER_LOG", "/tmp/carla_server.log")
        try:
            self._server_log_handle = open(server_log_path, "w")
            server_stdout = self._server_log_handle
            server_stderr = subprocess.STDOUT  # fold stderr into the same log
            print(f"CARLA server log: {server_log_path}")
        except OSError as log_err:
            print(f"Warning: could not open server log '{server_log_path}' ({log_err}); "
                  "discarding server output.")
            self._server_log_handle = None
            server_stdout = subprocess.DEVNULL
            server_stderr = subprocess.DEVNULL

        try:
            # Start without trying to monitor output to avoid interference
            self.process = subprocess.Popen(
                cmd,
                stdout=server_stdout,
                stderr=server_stderr,
                preexec_fn=os.setsid  # Create new process group
            )
        except Exception as e:
            raise Exception(f"Failed to start CARLA server: {e}")

        # Register cleanup handler once so we stop the server on normal exit
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

        print(f"CARLA server started (PID: {self.process.pid})")
        self.known_server_pid = self.process.pid
        # preexec_fn=os.setsid makes the wrapper the group leader, so its PID
        # is the pgid of the whole server tree (wrapper + UE4 binary).
        self.server_pgid = self.process.pid

        # Give CARLA time to start up without interference.
        initial_wait = 15
        if initial_wait > 0:
            print("Waiting for CARLA to initialize...")
            time.sleep(initial_wait)
        
        # Check if process is still running
        if self.process.poll() is not None:
            raise Exception(f"CARLA server exited during startup")
        
        return self.process
    
    def wait_for_server(self, timeout=120):
        """Wait for CARLA server to be ready"""
        if self.use_existing_server:
            return True
            
        print("Waiting for CARLA server RPC to be ready...")
        
        client = carla.Client('127.0.0.1', self.port)
        client.set_timeout(5.0)  # Shorter timeout for individual attempts
        
        start_time = time.time()
        last_error = None
        attempt_count = 0
        
        while time.time() - start_time < timeout:
            # Check if server process is still running
            if self.process and self.process.poll() is not None:
                raise Exception(f"CARLA server process died during startup")
            
            try:
                attempt_count += 1
                if attempt_count % 5 == 0:  # Every 10 seconds, show progress
                    elapsed = time.time() - start_time
                    print(f"Still waiting... ({elapsed:.0f}s elapsed, attempt {attempt_count})")
                
                version = client.get_server_version()
                print(f"CARLA server ready! Version: {version}")
                
                # Give it a moment more to fully stabilize
                time.sleep(2)
                return True
                
            except Exception as e:
                last_error = str(e)
                time.sleep(2)  # Wait between attempts
                print(".", end="", flush=True)
        
        print(f"\nTimeout waiting for CARLA server (waited {timeout}s)")
        if last_error:
            print(f"Last error: {last_error}")
        
        return False

    def _atexit_cleanup(self):
        """Internal handler to stop the server when the process exits."""
        cleanup_completed = [False]

        def _do_cleanup():
            try:
                self.stop_server()
                cleanup_completed[0] = True
            except KeyboardInterrupt:
                print("Cleanup interrupted by user")
                cleanup_completed[0] = True
            except Exception as e:
                print(f"Error during CARLA server shutdown: {e}")
                cleanup_completed[0] = True

        # Run cleanup in thread with timeout
        cleanup_thread = threading.Thread(target=_do_cleanup, daemon=True)
        cleanup_thread.start()
        cleanup_thread.join(timeout=15.0)

        if not cleanup_completed[0]:
            print("WARNING: CARLA server cleanup timed out after 15 seconds")

    def set_auto_stop_enabled(self, enabled):
        """Enable/disable automatic CARLA shutdown on process exit."""
        self.allow_auto_stop = enabled

    def get_known_server_pid(self):
        """Return PID of the CARLA server process if known."""
        return self.known_server_pid

    @staticmethod
    def _pgid_alive(pgid):
        """True while the group has a member that is not a zombie.

        A plain killpg(pgid, 0) probe is not enough: an orphaned CarlaUE4.sh
        wrapper (its launching VSE exited during a map-switch handoff) stays
        as an unreaped ZOMBIE after it dies, and zombies still count as group
        members -- the probe then reports a fully dead server as alive for as
        long as init takes to reap it (observed live 2026-07-07: ~20 s hang on
        exit + false 'survived SIGKILL' warning). Zombies hold no resources,
        so they are ignored here.
        """
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            # e.g. EPERM: the group exists but isn't ours -- treat as alive
            return True
        # Group exists -- check for at least one non-zombie member.
        # /proc/<pid>/stat: "pid (comm) state ppid pgrp ..." -- comm may
        # contain spaces/parens, so parse from the LAST ')'.
        pgid = int(pgid)
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                with open('/proc/%s/stat' % pid, 'rb') as f:
                    data = f.read()
                fields = data[data.rindex(b')') + 2:].split()
                state, pgrp = fields[0], int(fields[2])
            except (OSError, ValueError, IndexError):
                continue  # process vanished mid-scan or malformed
            if pgrp == pgid and state != b'Z':
                return True
        return False

    def _wait_for_group_exit(self, pgid, timeout):
        """Poll until the whole process group is gone; reap our own child."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None:
                self.process.poll()  # reap the wrapper when it exits
            if not self._pgid_alive(pgid):
                return True
            time.sleep(0.2)
        return not self._pgid_alive(pgid)

    def stop_server(self, force=False):
        """Stop CARLA server process"""
        if not force and not self.allow_auto_stop:
            print("Skipping CARLA server shutdown (handoff in progress)")
            return

        target_pid = None
        if self.process:
            target_pid = self.process.pid
        elif self.known_server_pid:
            target_pid = self.known_server_pid

        if target_pid:
            print("Stopping CARLA server...")
            # CarlaUE4.sh is only a wrapper: the UE4 binary is a child in the
            # same process group. Waiting on the wrapper alone lets the binary
            # outlive us, so we signal the GROUP and verify the GROUP is gone.
            pgid = None
            try:
                pgid = os.getpgid(target_pid)
            except OSError:
                # Group leader already exited; fall back to the pgid captured
                # at launch/adoption time (the group survives its leader).
                pgid = self.server_pgid
            if pgid is None:
                pgid = target_pid  # setsid leader == pgid; best effort

            try:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass  # group already gone (or unsignalable) - verify below
                stopped = self._wait_for_group_exit(pgid, self._stop_sigterm_wait_s)
                if not stopped:
                    logger.debug("SIGTERM did not stop server group %s, trying SIGKILL", pgid)
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
                    stopped = self._wait_for_group_exit(pgid, self._stop_sigkill_wait_s)
                if not stopped:
                    # Last resort: the same kill-everything sweep the startup
                    # path uses when the port is occupied.
                    print("CARLA server survived group kill - sweeping all CARLA processes...")
                    self.kill_existing_carla_processes()
                    stopped = not self._pgid_alive(pgid)
            except KeyboardInterrupt:
                print("CARLA server shutdown interrupted by user")
                raise
            self.process = None
            self.known_server_pid = None
            self.server_pgid = None
            self.use_existing_server = False
            self.allow_auto_stop = True
            if self._server_log_handle is not None:
                try:
                    self._server_log_handle.close()
                except OSError:
                    pass
                self._server_log_handle = None
            if stopped:
                print("CARLA server stopped")
            else:
                print(f"WARNING: CARLA server processes (pgid {pgid}) survived SIGKILL; "
                      "kill manually: pkill -f CarlaUE4")
        else:
            if self.use_existing_server:
                print("Leaving existing CARLA server running")

