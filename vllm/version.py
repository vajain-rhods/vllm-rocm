# SPDX-License-Identifier: Apache-2.0

try:
    __version__ = "0.7.3"
    __version_tuple__ = (0, 7, 3)

except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash:\n{e}",
                  RuntimeWarning,
                  stacklevel=2)

    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)
