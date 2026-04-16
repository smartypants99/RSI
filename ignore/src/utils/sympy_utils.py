"""Shared sympy helpers used by verifier and diagnostics.

Tolerant expression parsing (handles "^", "×", "÷", implicit multiplication,
LaTeX fragments like \\boxed{}, \\frac{}{}, \\sqrt{}), equation decision,
single-variable solving, numeric equivalence, answer normalization, and
equation extraction from prose.

All functions degrade gracefully when sympy is unavailable.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    import sympy  # type: ignore
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    _TRANSFORMS = (
        standard_transformations
        + (implicit_multiplication_application, convert_xor)
    )
    HAS_SYMPY = True
except Exception:  # pragma: no cover
    HAS_SYMPY = False
    _TRANSFORMS = None


# ── answer normalization ──

_RE_BOXED = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_RE_FRAC = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_RE_SQRT = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_RE_ANSWER_PREFIX = re.compile(
    r"^(?:the\s+)?(?:answer|result|solution|value)\s+is\s+",
    re.IGNORECASE,
)
_RE_DOLLAR = re.compile(r"^\$+|\$+$")
_RE_UNITS = re.compile(
    r"\s*(?:cm|mm|m|km|kg|g|mg|lb|oz|ft|in|s|sec|min|hr|hours?|days?|years?|"
    r"dollars?|euros?|cents?|percent|%|degrees?|°|radians?|rad)\s*$",
    re.IGNORECASE,
)

# Named constants to inject into the local dict for parse_expr
_SAFE_LOCALS = {}
if HAS_SYMPY:
    _SAFE_LOCALS = {
        "pi": sympy.pi,
        "Pi": sympy.pi,
        "e": sympy.E,
        "E": sympy.E,
        "i": sympy.I,
        "I": sympy.I,
        "inf": sympy.oo,
        "Inf": sympy.oo,
        "infinity": sympy.oo,
        "oo": sympy.oo,
        "sqrt": sympy.sqrt,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "log": sympy.log,
        "ln": sympy.ln,
        "exp": sympy.exp,
        "abs": sympy.Abs,
        "Abs": sympy.Abs,
    }


def normalize_answer(text: str) -> str:
    """Strip LaTeX wrappers, prose prefixes, units, and whitespace from an answer string."""
    if not text:
        return ""
    s = text.strip()
    # Extract from \boxed{...}
    m = _RE_BOXED.search(s)
    if m:
        s = m.group(1).strip()
    # Strip dollar-sign math delimiters
    s = _RE_DOLLAR.sub("", s).strip()
    # Strip "the answer is ..."
    s = _RE_ANSWER_PREFIX.sub("", s).strip()
    # Convert LaTeX fractions / sqrt
    s = _RE_FRAC.sub(r"((\1)/(\2))", s)
    s = _RE_SQRT.sub(r"sqrt(\1)", s)
    # Common LaTeX commands to plain
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi").replace("\\infty", "oo").replace("\\inf", "oo")
    s = s.replace("\\le", "<=").replace("\\ge", ">=")
    s = s.replace("\\left", "").replace("\\right", "")
    # Strip trailing units
    s = _RE_UNITS.sub("", s).strip()
    # Strip trailing punctuation
    s = s.rstrip(".!?,;:")
    return s.strip()


def sympify_safe(expr: str) -> Optional["sympy.Expr"]:
    """Tolerant parse_expr wrapper. Returns None on any failure."""
    if not HAS_SYMPY or not expr:
        return None
    cleaned = normalize_answer(expr)
    cleaned = cleaned.replace("^", "**").replace("×", "*").replace("÷", "/")
    # Remove remaining non-math characters but keep parens, operators, letters, digits
    cleaned = re.sub(r"[^\w\s+\-*/().=^<>,]", " ", cleaned)
    if not cleaned.strip() or "=" in cleaned:
        return None
    try:
        return parse_expr(cleaned, local_dict=_SAFE_LOCALS,
                          transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return None


def numeric_equiv(a: str, b: str, tol: float = 1e-9) -> Optional[bool]:
    """Check if two answer strings are mathematically equivalent.

    Tries symbolic simplification first, then numeric evaluation as fallback.
    Returns True/False if decidable, None if undecidable.
    """
    if not HAS_SYMPY:
        return None
    A = sympify_safe(a)
    B = sympify_safe(b)
    if A is None or B is None:
        return None
    try:
        diff = sympy.simplify(A - B)
        if diff == 0:
            return True
        # Try numeric evaluation for cases like pi vs 3.14159
        if not diff.free_symbols:
            try:
                return abs(complex(diff).real) < tol and abs(complex(diff).imag) < tol
            except (TypeError, ValueError, OverflowError):
                pass
        # Try .equals() which uses random numeric evaluation
        try:
            eq = A.equals(B)
            if eq is True:
                return True
            if eq is False:
                # Only trust False from .equals() if no free symbols
                if not diff.free_symbols:
                    return False
        except Exception:
            pass
        if diff.free_symbols:
            return None
        return False
    except Exception:
        return None


def equation_valid(lhs: str, rhs: str) -> Optional[bool]:
    """Return True/False if the equation can be decided, None if symbolic."""
    return numeric_equiv(lhs, rhs)


def solve_single_var(equations: list[tuple[str, str]]) -> dict[str, "sympy.Expr"]:
    """For each (lhs, rhs) with exactly one free symbol, solve for it.

    Returns {variable_name: value_expr}.
    """
    if not HAS_SYMPY:
        return {}
    out: dict[str, "sympy.Expr"] = {}
    for lhs, rhs in equations:
        L = sympify_safe(lhs)
        R = sympify_safe(rhs)
        if L is None or R is None:
            continue
        free = (L - R).free_symbols
        if len(free) != 1:
            continue
        var = next(iter(free))
        try:
            sols = sympy.solve(sympy.Eq(L, R), var)
            if sols:
                out[str(var)] = sols[0]
        except Exception:
            continue
    return out


def gather_equations(text: str) -> list[tuple[str, str]]:
    """Extract (lhs, rhs) equation pairs from prose text.

    Splits on sentence/colon/semicolon boundaries (preserving decimals),
    strips leading connective words and prose prefixes, then splits each
    clause on the first '='.
    """
    if not text:
        return []
    clauses = re.split(r"(?<!\d)\.(?!\d)|[:;!?\n]", text)
    out: list[tuple[str, str]] = []
    for clause in clauses:
        clause = clause.strip()
        if "=" not in clause:
            continue
        # Normalize "==" to "=" for code-style equations
        clause = clause.replace("==", "=")
        clause = re.sub(
            r"^(?:so|thus|therefore|hence|then|we\s+(?:have|get|obtain|find|know))\s+",
            "",
            clause, flags=re.IGNORECASE,
        ).strip()
        parts = clause.split("=")
        if len(parts) != 2:
            continue
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        rhs = re.split(r"[,]| and | so | therefore ", rhs, maxsplit=1)[0].strip()
        while True:
            m = re.match(r"^([A-Za-z]{3,})\s+(?=\S)", lhs)
            if not m:
                break
            lhs = lhs[m.end():]
        if lhs and rhs:
            out.append((lhs, rhs))
    return out
