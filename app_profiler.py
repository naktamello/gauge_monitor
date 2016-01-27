#!/usr/bin/env python
import pstats

# run "python -m cProfile -o profile_info gm_app.py" on shell first
p = pstats.Stats('./profile_info')
p.strip_dirs().sort_stats('tottime').print_stats(10)