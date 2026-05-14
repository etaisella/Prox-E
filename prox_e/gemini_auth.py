"""Optional internal auth for Gemini when GOOGLE_API_KEY is unset."""

_TORC_MISSING = (
    "GOOGLE_API_KEY is not set and the internal package 'torc' is not installed. "
    "Set GOOGLE_API_KEY for Gemini (see README), or on lab machines install the org "
    "'torc' helper used for Vertex workload identity."
)


def import_token_source_v2():
    """Return torc TokenSourceV2, or raise with a clear message if torc is absent."""
    try:
        from torc.torc_token_v2 import TokenSourceV2
    except ImportError as e:
        raise RuntimeError(_TORC_MISSING) from e
    return TokenSourceV2
