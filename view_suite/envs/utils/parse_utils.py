import re
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple


class ParsedAction(NamedTuple):
    name: str
    arg: Optional[str]


# --------------- Format registry ---------------

class FormatRegistry:
    """
    Registry that maps format names to parser functions.

    Usage::

        @FormatRegistry.register("free_think")
        def parse_free_think(response: str) -> Dict[str, Any]:
            ...

    Then call::

        result = FormatRegistry.parse("free_think", response)

    Every registered parser must return::

        {"ok": bool, "think": str, "actions_blob": str}
    """

    _registry: Dict[str, Callable[[str], Dict[str, Any]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator: register a parser function under *name*."""
        def decorator(fn: Callable[[str], Dict[str, Any]]):
            cls._registry[name] = fn
            return fn
        return decorator

    @classmethod
    def parse(cls, fmt: str, response: str) -> Dict[str, Any]:
        """Look up *fmt* in the registry and invoke the parser."""
        if fmt not in cls._registry:
            raise ValueError(
                f"Unknown format {fmt!r}. "
                f"Registered formats: {sorted(cls._registry)}"
            )
        return cls._registry[fmt](response)

    @classmethod
    def valid_formats(cls) -> set:
        """Return the set of currently registered format names."""
        return set(cls._registry)


# --------------- Format instruction templates ---------------

_FORMAT_TEMPLATES: Dict[str, str] = {
    "free_think": (
        "You need to think first, then answer, respond in this format:\n"
        "<think>[your reasoning here]</think><action>{action_example}</action>\n"
        "{action_description}"
    ),
    "eval_mode": (
        "You can think first, which is optional, then answer, respond in this format:\n"
        "[your reasoning here]<action>{action_example}</action>\n"
        "{action_description}"
    ),
    "no_think": (
        "You need to only give your answer, respond in this format:\n"
        "<action>{action_example}</action>\n"
        "{action_description}"
    ),
    # Grounding format: the agent must ground its decision in what it SEES, and
    # commit to a prediction of the next view. The <description> is goal-agnostic,
    # so unlike <think> it stays valid after a hindsight goal relabel -> it can be
    # cached on the graph node and reused as SFT supervision.
    "grounding": (
        "First briefly describe what you see, then reason, then act. "
        "Respond in this format:\n"
        "<description>[one short sentence: the main objects/layout in the current "
        "view]</description>"
        "<think>[one short sentence: why this action moves you toward the target]"
        "</think>"
        "<action>{action_example}</action>\n"
        "Keep <description> and <think> to ONE short sentence each, and ALWAYS end "
        "with the <action> tag.\n"
        "{action_description}"
    ),
}


def get_format_instruction(
    format_name: str,
    action_example: str = "answer(x)",
    action_description: str = "",
) -> str:
    """
    Return a human-readable format instruction string for *format_name*.

    The instruction tells the model which XML tags to use (``<think>``,
    ``<action>``) and embeds *action_example* as a placeholder inside
    ``<action>`` so the model knows the expected action syntax.

    Args:
        format_name: One of the registered format names
            (``"free_think"`` | ``"eval_mode"`` | ``"no_think"``).
        action_example: Environment-specific action placeholder, e.g.
            ``"answer(x)"`` for QA envs or ``"action_1|action_2|action_3|..."`` for
            tool envs.
        action_description: Optional description placed right after the
            ``<action>`` example line, e.g. ``"where x is A, B, C, or D."``.

    Raises:
        ValueError: If *format_name* is not registered.
    """
    if format_name not in _FORMAT_TEMPLATES:
        raise ValueError(
            f"Unknown format {format_name!r}. "
            f"Available: {sorted(_FORMAT_TEMPLATES)}"
        )
    desc = (action_description.strip() + "\n") if action_description.strip() else ""
    return _FORMAT_TEMPLATES[format_name].format(
        action_example=action_example,
        action_description=desc,
    )


# --------------- Regexes ---------------

# Strict: exactly <think>...</think><action>...</action>
_THINK_ACTION_RE = re.compile(
    r'^\s*<think>(?P<think>[\s\S]*?)</think>[ \t\r\n]*<action>(?P<action>[\s\S]*?)</action>\s*$',
    re.IGNORECASE
)

# Lenient: only requires <action>...</action> (think is optional)
_LENIENT_ACTION_RE = re.compile(
    r'<action>(?P<action>[\s\S]*?)</action>',
    re.IGNORECASE
)

# Strict action-only: the entire response must be exactly <action>...</action>
_STRICT_ACTION_ONLY_RE = re.compile(
    r'^\s*<action>(?P<action>[\s\S]*?)</action>\s*$',
    re.IGNORECASE
)


# --------------- Registered parsers ---------------

@FormatRegistry.register("free_think")
def parse_free_think(response: str) -> Dict[str, Any]:
    """
    Strict format: ``<think>...</think><action>...</action>``

    Returns ``{"ok": bool, "think": str, "actions_blob": str}``
    """
    m = _THINK_ACTION_RE.fullmatch(response)
    if not m:
        return {"ok": False, "think": "", "actions_blob": ""}
    return {
        "ok": True,
        "think": m.group("think").strip(),
        "actions_blob": m.group("action").strip(),
    }


# full grounding = description + think + action. <prediction> was dropped: a 3B model
# spent its whole budget on 4 tags and never emitted <action> (1992 of 2552 turns had
# NO action tag). The "next view" is simply the NEXT turn's <description>, which the
# graph builder already prefers, so nothing is lost.
_GROUNDING_RE = re.compile(
    r'^\s*<description>(?P<description>[\s\S]*?)</description>\s*'
    r'<think>(?P<think>[\s\S]*?)</think>\s*'
    r'<action>(?P<action>[\s\S]*?)</action>\s*$',
    re.IGNORECASE
)


@FormatRegistry.register("grounding")
def parse_grounding(response: str) -> Dict[str, Any]:
    """
    Grounding format:
    ``<description>..</description><think>..</think><prediction>..</prediction><action>..</action>``

    LENIENT on purpose. A small model emits sloppy variants (missing <prediction>,
    stray text between tags). If we required an exact fullmatch, every such turn
    would be a parse error -> no action executed, no reward, no learning signal, and
    the policy could never bootstrap into the format. So: we only REQUIRE a
    non-empty <action>; description/think/prediction are captured when present.
    ``full_format`` reports whether all four tags were present in the right order —
    that is what the per-turn format bonus should key on, so the format is still
    rewarded without breaking the episode.
    """
    a = _LENIENT_ACTION_RE.search(response)
    if not a or not a.group("action").strip():
        return {"ok": False, "description": "", "think": "", "prediction": "",
                "actions_blob": "", "full_format": False}

    def _grab(tag):
        m = re.search(rf"<{tag}>([\s\S]*?)</{tag}>", response, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    desc, think, pred = _grab("description"), _grab("think"), _grab("prediction")
    return {
        "ok": True,
        "description": desc,
        "think": think,
        "prediction": pred,
        "actions_blob": a.group("action").strip(),
        "full_format": bool(_GROUNDING_RE.fullmatch(response)),
    }


@FormatRegistry.register("eval_mode")
def parse_free_think_lenient(response: str) -> Dict[str, Any]:
    """
    Lenient format: only ``<action>...</action>`` required; ``<think>`` optional.

    Returns ``{"ok": bool, "think": str, "actions_blob": str}``
    """
    m = _LENIENT_ACTION_RE.search(response)
    if not m:
        return {"ok": False, "think": "", "actions_blob": ""}
    return {
        "ok": True,
        "think": "",
        "actions_blob": m.group("action").strip(),
    }


@FormatRegistry.register("no_think")
def parse_no_think(response: str) -> Dict[str, Any]:
    """
    No-think format: the response must be **exactly** ``<action>...</action>``
    (with optional surrounding whitespace). Any extra content makes ``ok=False``.

    Returns ``{"ok": bool, "think": str, "actions_blob": str}``
    """
    m = _STRICT_ACTION_ONLY_RE.fullmatch(response)
    if m:
        return {"ok": True, "think": "", "actions_blob": m.group("action").strip()}
    return {"ok": False, "think": "", "actions_blob": response.strip()}


# --------------- Action parsing helpers ---------------

def _normalize_ws(s: str) -> str:
    """Collapse all whitespace into a single space and strip ends."""
    return re.sub(r"\s+", " ", s.strip())


def parse_actions(
    actions_blob: str,
    sep: str = "|",
    allow_trailing_empty: bool = True,
) -> Tuple[bool, List[ParsedAction]]:
    r"""
    Parse action strings like "foo|bar(arg)|baz".

    Each token must be one of:
      - name(arg)   -> name matches [A-Za-z_]\\w*, arg cannot contain parentheses
      - name        -> no-arg action

    Returns:
      (ok, actions)
        ok: bool                # True if all tokens parsed successfully
        actions: List[ParsedAction]
    """
    normalized = re.sub(r"\s+", "", actions_blob) if sep == "|" else actions_blob
    parts = normalized.split(sep)
    tokens = [t for t in (x.strip() for x in parts) if t or not allow_trailing_empty]
    actions: List[ParsedAction] = []

    for tok in tokens:
        # Case 1: name(arg), where arg excludes parentheses
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*\(\s*([^()]*)\s*\)", tok)
        if m:
            actions.append(
                ParsedAction(name=m.group(1).lower(), arg=_normalize_ws(m.group(2)))
            )
            continue

        # Case 2: name (no arguments)
        m = re.fullmatch(r"([A-Za-z_]\w*)", tok)
        if m:
            actions.append(ParsedAction(name=m.group(1).lower(), arg=None))
            continue

        # Invalid token
        return False, []

    return True, actions


# --------------- High-level dispatch ---------------

def parse_no_tool_action_str(action_str: str, format: str = "free_think") -> Tuple[bool, str]:
    """
    Parse a no-tool answer string, expected format:
        <think>...</think><action>answer(some_text)</action>

    Rules:
      - Format dispatch via :class:`FormatRegistry`
      - Must contain exactly one action
      - The action must be "answer" with a non-empty argument

    Args:
      action_str: The raw response to parse.
      format: One of the registered format names
              (``"free_think"`` | ``"eval_mode"`` | ``"no_think"``).

    Returns:
      (format_ok, answer_arg)
    """
    ft = FormatRegistry.parse(format, action_str)

    if not ft["ok"]:
        return False, ""

    actions_ok, parsed_actions = parse_actions(
        ft["actions_blob"], sep="|", allow_trailing_empty=True
    )
    if not actions_ok or len(parsed_actions) != 1:
        return False, ""

    a = parsed_actions[0]
    if a.name != "answer" or not a.arg or not a.arg.strip():
        return False, ""

    return True, a.arg
