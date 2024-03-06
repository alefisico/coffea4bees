from functools import cache


@cache
def fsspec_read(path: str) -> str:
    import fsspec
    with fsspec.open(path, 'rt') as f:
        return f.read()
