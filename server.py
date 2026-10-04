# -*- coding: utf-8 -*-
"""
TennisAid 本地服务（最小实现，仅用标准库）。
启动后在本地端口起一个 HTTP 服务，自动打开浏览器显示网页界面。
打包成 exe 后即为完整桌面应用。
"""
import os
import sys
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

from MyTennisAid import pipeline

PORT = 7860
JOBS = {}
JOBS_LOCK = threading.Lock()
_job_counter = 0

# 本地访问不走系统代理
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])


def web_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(getattr(sys, "_MEIPASS", "."), "web")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            self.send_error(404)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._file(os.path.join(web_dir(), "index.html"), "text/html; charset=utf-8")
        if u.path == "/status":
            jid = parse_qs(u.query).get("job", [None])[0]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if not job:
                return self._json({"error": "未知任务"}, 404)
            return self._json({k: job[k] for k in ("progress", "msg", "done", "result", "error", "clips", "clips_dir")})
        if u.path.startswith("/clips/"):
            parts = u.path[len("/clips/"):].split("/", 1)
            if len(parts) == 2:
                jid, fname = parts
                fname = unquote(os.path.basename(fname))  # 中文文件名需解码
                with JOBS_LOCK:
                    job = JOBS.get(jid)
                if job and job.get("clips_dir"):
                    p = os.path.join(job["clips_dir"], fname)
                    if os.path.isfile(p):
                        return self._file(p, "video/mp4")
        self.send_error(404)

    def do_POST(self):
        if self.path == "/analyze":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                video = body.get("path", "").strip().strip('"')
            except Exception:
                return self._json({"error": "请求格式错误"}, 400)
            if not video or not os.path.exists(video):
                return self._json({"error": "找不到视频文件：%s" % video}, 400)

            global _job_counter
            with JOBS_LOCK:
                _job_counter += 1
                jid = str(_job_counter)
                JOBS[jid] = {"progress": 0.0, "msg": "准备中…", "done": False,
                             "result": None, "error": None, "clips_dir": None, "clips": []}
            t = threading.Thread(target=_run_job, args=(jid, video), daemon=True)
            t.start()
            return self._json({"job": jid})

        if self.path == "/shutdown":
            self._json({"ok": True})
            threading.Thread(target=lambda: os._exit(0), daemon=True).start()
            return
        self.send_error(404)

    def log_message(self, *a):
        pass  # 静默，不刷屏


def _run_job(jid, video):
    def cb(frac, msg):
        with JOBS_LOCK:
            JOBS[jid]["progress"] = round(frac * 100)
            JOBS[jid]["msg"] = msg
    try:
        stem = os.path.splitext(os.path.basename(video))[0]
        # 输出目录在视频同目录下，仅存放最终 mp4 片段
        out_dir = os.path.join(os.path.dirname(os.path.abspath(video)), stem + "_MyTennisAid")
        segs = pipeline.process_video(video, progress=cb)
        clips = pipeline.cut_clips(video, segs, out_dir) if segs else []
        with JOBS_LOCK:
            JOBS[jid]["result"] = segs
            JOBS[jid]["clips_dir"] = out_dir if segs else None
            JOBS[jid]["clips"] = [os.path.basename(c) for c in clips]
            JOBS[jid]["done"] = True
            JOBS[jid]["progress"] = 100
            JOBS[jid]["msg"] = "完成"
    except Exception as e:
        with JOBS_LOCK:
            JOBS[jid]["error"] = str(e)
            JOBS[jid]["done"] = True


def main():
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        _alert("启动失败：端口 %d 被占用或无法监听。\n%s" % (PORT, e))
        return
    url = "http://127.0.0.1:%d" % PORT
    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    print("TennisAid 已启动：", url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def _alert(msg):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "TennisAid", 0x10)
    except Exception:
        print(msg)


if __name__ == "__main__":
    main()
