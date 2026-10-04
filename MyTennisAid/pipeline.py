# -*- coding: utf-8 -*-
"""
MyTennisAid 核心处理管线（可复用，供命令行和网页 UI 调用）。

process_video(video_path, progress=None) -> list[dict]
  返回最终精彩片段列表，每项含 start/end/hits/duration/motion_ratio。
  中间产物（音轨、击球点、代理视频、运动量）全部写入临时目录，处理完自动删除，
  最终只把剪辑好的 mp4 片段写到指定输出目录。
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import cv2
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, find_peaks
from scipy.ndimage import median_filter

# ---- 默认参数（阈值经真实视频校准）----
CFG = {
    "audio_sr": 24000,        # 音轨重采样率
    "hp_cutoff": 300,         # 高通截止频率(Hz)
    "win_ms": 20,             # 短时能量窗长
    "hop_ms": 10,             # 帧移
    "min_hit_gap": 0.30,      # 两拍最小间隔(s)
    "rally_gap": 4.0,         # 回合断开间隔(s)
    "min_hits": 6,            # 候选回合最少拍数
    "threshold_ratio": 4.0,   # 击球能量阈值倍数
    "floor_ratio": 0.02,      # 能量绝对下限
    "proxy_fps": 6,           # 运动分析代理视频帧率
    "proxy_scale": 320,       # 代理视频宽度
    "motion_ratio": 1.30,     # 运动量门控阈值（>此值判为本方回合）
    "pad_start": 2.0,         # 输出片段向前多取(s)
    "pad_end": 2.0,           # 输出片段向后多取(s)
}


def _ffmpeg():
    """定位 ffmpeg 二进制。

    优先级：打包成 exe 后的资源目录/同级目录 → 项目根目录的 ffmpeg.exe →
    开发环境兜底用 pip 安装的 imageio-ffmpeg 自带的 ffmpeg。
    """
    import sys
    cands = []
    if getattr(sys, "frozen", False):  # PyInstaller 打包后
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        cands += [os.path.join(base, "ffmpeg.exe"),
                  os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe")]
    else:  # 开发环境：pipeline.py 在 MyTennisAid/ 下，项目根目录在其上一级
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cands += [os.path.join(root, "ffmpeg.exe"),
                  os.path.join(root, "bin", "ffmpeg.exe")]
    for c in cands:
        if os.path.exists(c):
            return c
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise FileNotFoundError(
        "找不到 ffmpeg。请安装 ffmpeg，或执行 `pip install imageio-ffmpeg`")


def _run(cmd):
    subprocess.run(cmd, check=True)


def process_video(video_path, progress=None, **kw):
    """完整流程。progress 为可选回调 progress(0~1, '描述')。

    所有中间文件写入临时目录，函数返回前自动清理，不残留任何中间产物。
    """
    cfg = {**CFG, **kw}
    tmp = tempfile.mkdtemp(prefix="mytinnisaid_")

    def report(frac, msg):
        if progress:
            progress(frac, msg)
        else:
            print("[MyTennisAid] %s" % msg)

    try:
        # 1. 提取音轨
        wav = os.path.join(tmp, "audio.wav")
        report(0.02, "提取音轨…")
        _run([_ffmpeg(), "-y", "-loglevel", "error", "-i", video_path,
              "-vn", "-ac", "1", "-ar", str(cfg["audio_sr"]),
              "-c:a", "pcm_s16le", wav])

        # 2. 检测击球点
        report(0.05, "检测击球点…")
        sr, x = wavfile.read(wav)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float32) / 32768.0
        sos = butter(4, cfg["hp_cutoff"] / (sr / 2), "high", output="sos")
        y = sosfilt(sos, x)
        win = int(cfg["win_ms"] * sr / 1000)
        hop = int(cfg["hop_ms"] * sr / 1000)
        n = (len(y) - win) // hop + 1
        energy = np.empty(n, dtype=np.float32)
        for i in range(n):
            seg = y[i * hop:i * hop + win]
            energy[i] = np.sqrt(np.mean(seg * seg))
        bg_win = int(5.0 * 1000 / cfg["hop_ms"])
        if bg_win % 2 == 0:
            bg_win += 1
        bg = median_filter(energy, size=bg_win, mode="nearest")
        floor = cfg["floor_ratio"] * np.median(energy)
        thr = bg * cfg["threshold_ratio"] + floor
        min_dist = int(cfg["min_hit_gap"] * 1000 / cfg["hop_ms"])
        peaks, _ = find_peaks(energy, height=thr, distance=min_dist)
        hits = (peaks * hop + win // 2) / sr
        report(0.15, "检测到 %d 个击球点" % len(hits))

        # 3. 聚类候选回合
        rallies = []
        if len(hits) > 0:
            splits = np.where(np.diff(hits) > cfg["rally_gap"])[0] + 1
            for g in np.split(hits, splits):
                if len(g) >= cfg["min_hits"]:
                    rallies.append({"start": float(g[0]), "end": float(g[-1]),
                                    "hits": int(len(g))})
        report(0.20, "候选回合 %d 个" % len(rallies))
        if not rallies:
            return []

        # 4. 抽代理视频 + 算画面运动量
        proxy = os.path.join(tmp, "proxy.mp4")
        report(0.25, "抽取代理视频（用于画面运动量分析）…")
        _run([_ffmpeg(), "-y", "-loglevel", "error", "-noautorotate", "-i", video_path,
              "-vf", "fps=%d,scale=%d:-2" % (cfg["proxy_fps"], cfg["proxy_scale"]),
              "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", proxy])

        report(0.35, "计算画面运动量…")
        cap = cv2.VideoCapture(proxy)
        fps = cap.get(cv2.CAP_PROP_FPS) or cfg["proxy_fps"]
        t, m = [], []
        prev = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            m.append(0.0 if prev is None else float(cv2.absdiff(gray, prev).mean()))
            t.append(len(t) / fps)
            prev = gray
        cap.release()
        t = np.array(t, dtype=np.float64)
        m = np.array(m, dtype=np.float32)
        report(0.55, "画面运动量计算完成")

        # 5. 运动量门控：本方回合（球员跑动）运动量大，隔壁串音（球员站着）运动量小
        baseline = float(np.median(m))
        kept = []
        for r in rallies:
            mask = (t >= r["start"] - 1.0) & (t <= r["end"] + 1.0)
            r["motion"] = float(np.mean(m[mask])) if mask.sum() else 0.0
            r["motion_ratio"] = r["motion"] / baseline if baseline > 0 else 0.0
            if r["motion_ratio"] >= cfg["motion_ratio"]:
                kept.append(r)
        kept.sort(key=lambda r: r["start"])
        report(0.80, "画面门控后保留 %d 个精彩片段" % len(kept))

        # 6. 加 padding 输出
        result = []
        for r in kept:
            s = max(0.0, r["start"] - cfg["pad_start"])
            e = r["end"] + cfg["pad_end"]
            result.append({"start": round(s, 2), "end": round(e, 2),
                           "hits": r["hits"], "duration": round(e - s, 2),
                           "avg_gap": round((r["end"] - r["start"]) / max(1, r["hits"] - 1), 2),
                           "motion_ratio": round(r["motion_ratio"], 2)})
        report(1.0, "完成")
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # 清理所有中间文件


def cut_clips(video_path, segments, out_dir):
    """把片段流复制剪辑成 mp4（仅写 mp4 到 out_dir），返回文件路径列表。"""
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for i, s in enumerate(segments):
        fn = os.path.join(out_dir, "精彩%02d_%02d分%02d秒_%d拍.mp4" %
                          (i + 1, int(s["start"]) // 60, int(s["start"]) % 60, s["hits"]))
        if not os.path.exists(fn):
            _run([_ffmpeg(), "-y", "-loglevel", "error", "-ss", str(s["start"]),
                  "-to", str(s["end"]), "-i", video_path, "-c", "copy", fn])
        files.append(fn)
    return files
