def autopad(k, p=None):
    """
    Pad to 'same' shape outputs.
    """
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
