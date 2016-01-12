#!/usr/bin/env python
import os.path
import os, platform, subprocess, re
import gauge_det

subprocess.call("~/rpi-fbcp/build/fbcp", shell=True)
while(1):
    try:
        result = gauge_det.main()
        if result and os.path.exists("./canvas.jpg"):
            subprocess.call("sudo fbi -T 1 canvas.jpg", shell=True)
    except Exception:
        print "error running script"