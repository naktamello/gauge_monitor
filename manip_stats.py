#!/usr/bin/env python
import sys
import io
from contextlib import contextmanager
import pstats

p = pstats.Stats('profile_out.txt')
f= io.StringIO()
@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout

with stdout_redirector(f):
    p.strip_dirs().sort_stats('time').print_stats()