"""Shared sandboxed code runner for untrusted Python snippets.

Spawns an isolated `python -I` subprocess with tight RLIMITs, a wall-clock
timeout, a scrubbed environment, an ephemeral cwd, and an audit-hook based
import/syscall blocklist. Returns (ok, tail) where `tail` is stdout on
success or stderr on failure (last ~500 chars).
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap

_SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ASIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r'"private_key_id"\s*:\s*"[A-Fa-f0-9]{20,}"'),
    re.compile(r"(?i)DefaultEndpointsProtocol=[^;\s]+;AccountKey=[A-Za-z0-9+/=]{20,}"),
    re.compile(r"""(?i)api[_-]?key\s*[:=]\s*['"][A-Za-z0-9_\-]{16,}['"]"""),
]

# Maximum stdout/stderr chars to keep before truncating (prevents print floods)
_MAX_OUTPUT_CHARS = 64 * 1024  # 64 KiB


def _scrub(s: str) -> str:
    """Redact secrets and sanitize non-ASCII from output."""
    s = s.encode("ascii", "replace").decode("ascii")
    for pat in _SECRET_PATTERNS:
        s = pat.sub("[REDACTED]", s)
    return s


_PRELUDE = textwrap.dedent(
    """
    import resource, sys, builtins

    # ── Resource limits ──
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    try:
        resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, ({fsize}, {fsize}))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, OSError, AttributeError):
        pass
    sys.setrecursionlimit(1000)

    # ── Module blocklist ──
    _BLOCKED_MODULES = {{
        "socket", "ssl", "urllib", "urllib.request", "urllib3", "http",
        "http.client", "ftplib", "telnetlib", "smtplib", "poplib",
        "imaplib", "requests", "httpx", "asyncio", "subprocess",
        "multiprocessing", "ctypes", "cffi", "importlib", "signal",
    }}

    # ── Syscall / audit-event blocklist ──
    _BLOCKED_EVENTS = {{
        "socket.connect", "socket.bind", "socket.gethostbyname",
        "urllib.Request", "subprocess.Popen", "os.exec", "os.fork",
        "os.forkpty", "os.spawn", "os.system", "os.posix_spawn",
        "shutil.copyfile", "shutil.move",
        "os.chdir", "os.chmod", "os.chown", "os.symlink", "os.link",
        "os.rename", "os.remove", "os.unlink", "os.rmdir", "os.truncate",
        "os.putenv", "os.unsetenv",
        "code.__new__",
    }}

    import os as _os_mod
    _WORKDIR_PREFIX = _os_mod.path.realpath(_os_mod.getcwd())
    if not _WORKDIR_PREFIX.endswith(_os_mod.sep):
        _WORKDIR_PREFIX = _WORKDIR_PREFIX + _os_mod.sep

    # ── Disable dangerous builtins ──
    _real_import = builtins.__import__

    def _safe_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in _BLOCKED_MODULES or top in _BLOCKED_MODULES:
            raise PermissionError(f"import blocked: {{name}}")
        return _real_import(name, *args, **kwargs)

    builtins.__import__ = _safe_import

    # Neuter eval/exec/compile so even indirect calls fail
    _orig_eval = builtins.eval
    _orig_exec = builtins.exec

    def _blocked_eval(*a, **kw):
        raise PermissionError("eval() is blocked in sandbox")

    def _blocked_exec(*a, **kw):
        raise PermissionError("exec() is blocked in sandbox")

    builtins.eval = _blocked_eval
    builtins.exec = _blocked_exec

    # ── Audit hook ──
    def _audit(event, args):
        if event == "import":
            name = args[0] if args else ""
            if name in _BLOCKED_MODULES or name.split(".")[0] in _BLOCKED_MODULES:
                raise PermissionError(f"import blocked: {{name}}")
        if event in _BLOCKED_EVENTS:
            raise PermissionError(f"blocked syscall: {{event}}")
        if event == "open":
            path = args[0] if args else ""
            if isinstance(path, (bytes, bytearray)):
                try:
                    path = path.decode()
                except Exception:
                    raise PermissionError("open: non-decodable path")
            if isinstance(path, int):
                return
            if isinstance(path, str):
                mode = args[1] if len(args) > 1 else "r"
                if isinstance(mode, str) and any(c in mode for c in "wxa+"):
                    raise PermissionError(f"open for writing blocked: {{path}}")
                abs_p = _os_mod.path.realpath(path) if path else ""
                if not abs_p.startswith(_WORKDIR_PREFIX):
                    raise PermissionError(f"open outside workdir: {{path}}")
            else:
                raise PermissionError("open: non-string path")

    sys.addaudithook(_audit)

    # Block sys.exit so snippets can't silently exit with code 0
    def _blocked_exit(*a, **kw):
        raise PermissionError("sys.exit() is blocked in sandbox")
    sys.exit = _blocked_exit

    # Clean up os module from user namespace
    del _os_mod
    """
).strip()


def _preexec():
    """Detach child into its own session for clean kill-group."""
    try:
        os.setsid()
    except OSError:
        pass


def run_python_sandboxed(
    source: str, timeout_s: int = 5, memory_mb: int = 256
) -> tuple[bool, str]:
    """Execute `source` in an isolated Python subprocess.

    Returns (ok, tail). `ok` is True iff the subprocess exited 0.
    `tail` is the last ~500 characters of stdout (on success) or
    stderr/stdout (on failure), suitable for inclusion in diagnostics.
    """
    timeout_s = max(1, int(timeout_s))
    memory_mb = max(64, int(memory_mb))
    prelude = _PRELUDE.format(
        cpu=timeout_s,
        mem=memory_mb * 1024 * 1024,
        fsize=1024 * 1024,
    )
    full = prelude + "\n" + source
    workdir = tempfile.mkdtemp(prefix="sbx_")
    path = os.path.join(workdir, "snippet.py")
    with open(path, "w") as f:
        f.write(full)
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workdir,
            env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "", "HOME": workdir},
            start_new_session=True,
            preexec_fn=_preexec if sys.platform != "win32" else None,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s + 1)
        except subprocess.TimeoutExpired:
            if sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                proc.kill()
            try:
                proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            return False, f"timeout after {timeout_s}s"

        # Truncate excessive output (print floods)
        if stdout and len(stdout) > _MAX_OUTPUT_CHARS:
            stdout = stdout[:_MAX_OUTPUT_CHARS] + "\n... [truncated]"
        if stderr and len(stderr) > _MAX_OUTPUT_CHARS:
            stderr = stderr[:_MAX_OUTPUT_CHARS] + "\n... [truncated]"

        if proc.returncode == 0:
            return True, _scrub(stdout or "")[-500:]
        return False, _scrub(stderr or stdout or "")[-500:]
    except Exception as e:
        return False, f"sandbox error: {type(e).__name__}: {e}"
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except OSError:
            pass
