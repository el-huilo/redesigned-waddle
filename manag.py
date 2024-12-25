import subprocess
import os

if __name__ == "__main__":
    p = subprocess.Popen(["/usr/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:7860"])
    grad = subprocess.Popen(["python", "/content/redesigned-waddle/app.py"])
    outs, errs = p.communicate()
    print(outs)