class ProviderOverloadedError(Exception):
    """
    The provider is saturated. The request is valid and would succeed later,
    so this maps to HTTP 503 with a Retry-After header.

    Only for conditions that clear on their own. A model that cannot fit in
    memory is not one of them — retrying that never helps, so it stays a
    ValueError (400).
    """

    retry_after = 60
