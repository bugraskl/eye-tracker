"""The opt-in update check and installer (the only networking code of the app).

Everything that decides what may be fetched, and what may be installed, is plain
Python (:mod:`.release`, :mod:`.fetch`, :mod:`.installer`) and tested without a
network. Only :mod:`.winhttp` talks to the network, through Windows' own WinHTTP.
See ``docs/privacy.md`` for what the check sends and when.
"""
