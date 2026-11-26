#!/usr/bin/env python3
"""Export presentation to PDF using Chrome screenshots → PIL → PDF.
Each slide is captured as a 3840x2160 screenshot (2x Retina scale), then combined into a PDF."""

import subprocess
import os
import sys
from pathlib import Path

PRESENTATION = Path(__file__).parent / "index.html"
OUTPUT_PDF = Path(__file__).parent / "RoadVision_Presentation.pdf"

# Ensure Pillow is available
try:
    from PIL import Image
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "Pillow"], check=True)
    from PIL import Image

try:
    LANCZOS_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_FILTER = Image.LANCZOS

html_content = PRESENTATION.read_text(encoding="utf-8")
slide_count = html_content.count('data-slide="')
print(f"Found {slide_count} slides")

screenshots = []
temp_html_files = []

# Define target high resolution dimensions (2x scale factor of 1920x1080)
TARGET_WIDTH = 3840
TARGET_HEIGHT = 2160

try:
    for i in range(slide_count):
        # Create a standalone HTML that shows only slide i, filling the viewport
        inject_css = f"""
        <style>
        html, body {{
            overflow: hidden !important;
            width: 1920px !important;
            height: 1080px !important;
            margin: 0 !important;
            padding: 0 !important;
            background-color: #0a0a0f !important;
        }}
        .slides-wrapper {{
            overflow: hidden !important;
            width: 1920px !important;
            height: 1080px !important;
        }}
        .slide {{
            position: absolute !important;
            width: 1920px !important;
            height: 1080px !important;
            min-height: 1080px !important;
            max-height: 1080px !important;
            display: none !important;
            opacity: 1 !important;
            transform: none !important;
            pointer-events: all !important;
            overflow: hidden !important;
        }}
        .slide[data-slide="{i}"] {{
            display: flex !important;
        }}
        .nav-bar, .kbd-hint, .fullscreen-btn, .progress-bar, .slide-dots,
        .slide-counter, #slideDots {{ display: none !important; }}
        </style>
        """

        single_html = html_content.replace("</head>", inject_css + "</head>")
        
        # Write to the same directory as PRESENTATION to resolve relative paths
        slide_html_path = PRESENTATION.parent / f"_tmp_slide_{i:02d}.html"
        slide_html_path.write_text(single_html, encoding="utf-8")
        temp_html_files.append(slide_html_path)

        screenshot_path = PRESENTATION.parent / f"_tmp_slide_{i:02d}.png"
        screenshots.append(screenshot_path)

        # Chrome headless flags for high quality capture
        cmd = [
            "google-chrome",
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-software-rasterizer",
            "--hide-scrollbars",
            "--window-size=1920,1080",
            "--force-device-scale-factor=2", # 2x scale factor for crisp rendering (3840x2160)
            "--virtual-time-budget=5000",      # Give Chrome 5 seconds of virtual time to render fully
            f"--screenshot={screenshot_path}",
            f"file://{slide_html_path.resolve()}",
        ]

        print(f"  Capturing slide {i+1}/{slide_count}...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, timeout=45)
        if screenshot_path.exists():
            print(f"✓ ({screenshot_path.stat().st_size // 1024} KB)")
        else:
            print("✗ FAILED")
            if result.stderr:
                print(result.stderr.decode()[-300:])
            if result.stdout:
                print(result.stdout.decode()[-300:])

    # Convert screenshots to PDF
    print(f"\nConverting {len(screenshots)} screenshots to PDF...")

    images = []
    first_img = None

    for idx, ss_path in enumerate(screenshots):
        if not ss_path.exists():
            continue
        img = Image.open(ss_path).convert("RGB")
        # Ensure exact TARGET_WIDTH x TARGET_HEIGHT
        if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
            img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), LANCZOS_FILTER)
        
        if idx == 0 or first_img is None:
            first_img = img
        else:
            images.append(img)

    if first_img:
        # Save as PDF: 3840x2160 pixels at 192 DPI = 20" x 11.25" (widescreen landscape)
        first_img.save(
            str(OUTPUT_PDF),
            "PDF",
            resolution=192.0,
            save_all=True,
            append_images=images,
        )
        print(f"\n✅ PDF saved: {OUTPUT_PDF}")
        print(f"   Size: {OUTPUT_PDF.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   Pages: {len(images) + 1}")
        print(f"   Dimensions: 3840×2160 px per page (High Res 2K/4K Widescreen)")
    else:
        print("❌ No screenshots were captured successfully!")

finally:
    # Cleanup temp files
    print("\nCleaning up temporary files...")
    for f in temp_html_files:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass
    for f in screenshots:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass
