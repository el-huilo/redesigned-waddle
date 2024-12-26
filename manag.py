import subprocess
import sys

if __name__ == "__main__":
    try:
        mode = sys.argv[1]
    except:
        mode = "?__theme=dark"
    else:
        if mode == 'dark':
            mode = "?__theme=dark"
        else:
            mode = "?__theme=light"
    p = subprocess.Popen(["/usr/bin/cloudflared", "tunnel", "--url", f"http://127.0.0.1:7860/{mode}"])
    try:
        mode = sys.argv[2]
    except:
        grad = subprocess.Popen(["python", "/content/redesigned-waddle/app.py"])
    else:
        grad = subprocess.Popen(["python", "/content/redesigned-waddle/app.py", f"{mode}"])
    outs, errs = p.communicate()
    print(outs)