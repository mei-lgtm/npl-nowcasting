#!/usr/bin/env python3
"""Record a walkthrough demo video of the SINAR web app."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
FRAMES = OUT / "raw"
VID_DIR = OUT / "recordings"
BASE = "http://127.0.0.1:8000/"
W, H = 1440, 900

# (nav page id, dwell seconds, optional JS after navigate)
SCENES = [
    ("dashboard", 3.2, None),
    ("data", 3.5, None),
    ("indicators", 3.2, None),
    ("models", 4.0, None),
    ("scenario", 4.5, """
        const a = document.querySelector('input[type=range][data-sc]');
        if (a) {
          const mid = (+a.min + +a.max) / 2;
          a.value = mid;
          a.dispatchEvent(new Event('input', {bubbles:true}));
          a.dispatchEvent(new Event('change', {bubbles:true}));
        }
    """),
    ("policy", 4.0, None),
]


def ensure_server():
    import urllib.request
    try:
        with urllib.request.urlopen(BASE, timeout=3) as r:
            if r.status == 200:
                return
    except Exception:
        pass
    raise SystemExit(
        "SINAR server tidak merespons di http://127.0.0.1:8000/ — jalankan uvicorn dulu."
    )


def make_title_png(path: Path, title: str, subtitle: str = ""):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, H), (12, 31, 58))
    draw = ImageDraw.Draw(img)
    # soft gradient bars
    for i in range(H):
        t = i / H
        r = int(12 + 20 * t)
        g = int(31 + 40 * t)
        b = int(58 + 30 * t)
        draw.line([(0, i), (W, i)], fill=(r, g, b))
    # accent line
    draw.rectangle([0, H // 2 - 90, W, H // 2 - 86], fill=(37, 99, 235))
    try:
        font_lg = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 56)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 26)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = font_lg

    def center(text, y, font, fill=(255, 255, 255)):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) / 2, y), text, font=font, fill=fill)

    center(title, H // 2 - 70, font_lg, (255, 255, 255))
    if subtitle:
        center(subtitle, H // 2 + 10, font_sm, (147, 197, 253))
    img.save(path)


def png_to_clip(png: Path, seconds: float, out_mp4: Path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(png),
            "-c:v", "libx264", "-t", str(seconds), "-pix_fmt", "yuv420p",
            "-vf", f"scale={W}:{H}", "-r", "30",
            str(out_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def webm_to_mp4(webm: Path, mp4: Path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(webm),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", f"scale={W}:{H}", "-r", "30",
            str(mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def concat_mp4(parts: list[Path], out: Path):
    lst = OUT / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(lst), "-c", "copy", str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def record_walkthrough() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    FRAMES.mkdir(parents=True)
    VID_DIR.mkdir(parents=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": W, "height": H},
            device_scale_factor=1,
            record_video_dir=str(VID_DIR),
            record_video_size={"width": W, "height": H},
        )
        page = context.new_page()
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_selector("#nav button", timeout=30000)
        time.sleep(1.0)

        # Load demo pipeline so charts populate
        page.evaluate(
            """async () => {
              if (typeof runPipeline === 'function') {
                await runPipeline({forceDemo: true, fromMethod: true});
              }
            }"""
        )
        # wait until dashboard KPI fills or timeout
        try:
            page.wait_for_function(
                """() => {
                  const v = document.getElementById('dkNpl');
                  return v && v.textContent && v.textContent.trim() !== '—';
                }""",
                timeout=90000,
            )
        except Exception:
            pass
        time.sleep(1.5)

        # Brief pause on dashboard (already there)
        time.sleep(2.0)

        for pid, dwell, js in SCENES:
            btn = page.locator(f'#nav button[data-p="{pid}"]')
            if btn.count():
                btn.first.click()
                page.wait_for_timeout(600)
            if js:
                try:
                    page.evaluate(js)
                    page.wait_for_timeout(500)
                except Exception:
                    pass
            # gentle scroll to show lower content on longer pages
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            page.wait_for_timeout(int(dwell * 500))
            page.evaluate("window.scrollTo({top: Math.min(420, document.body.scrollHeight), behavior: 'smooth'})")
            page.wait_for_timeout(int(dwell * 500))

        time.sleep(1.0)
        video = page.video
        page.close()
        webm = Path(video.path()) if video else None
        context.close()
        browser.close()

    if not webm or not webm.exists():
        # find newest webm in VID_DIR
        webs = sorted(VID_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not webs:
            raise SystemExit("Rekaman video Playwright tidak ditemukan.")
        webm = webs[0]
    return webm


def main():
    ensure_server()
    print("Merekam walkthrough SINAR…")
    webm = record_walkthrough()
    print("WebM:", webm)

    walk = OUT / "walkthrough.mp4"
    webm_to_mp4(webm, walk)

    intro_png = OUT / "intro.png"
    outro_png = OUT / "outro.png"
    make_title_png(
        intro_png,
        "SINAR",
        "Sistem Nowcasting dan Analisis Risiko NPL Konsumsi",
    )
    make_title_png(
        outro_png,
        "Terima kasih",
        "Demo SINAR · Bank Indonesia",
    )
    intro = OUT / "intro.mp4"
    outro = OUT / "outro.mp4"
    png_to_clip(intro_png, 2.5, intro)
    png_to_clip(outro_png, 2.2, outro)

    final = ROOT / "SINAR-demo.mp4"
    concat_mp4([intro, walk, outro], final)
    # also copy into demo/
    shutil.copy2(final, OUT / "SINAR-demo.mp4")
    size_mb = final.stat().st_size / (1024 * 1024)
    print(f"Selesai: {final} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
