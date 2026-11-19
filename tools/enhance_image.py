from PIL import Image, ImageFilter
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "diagrams", "methodology.png")

if not os.path.exists(IN_PATH):
    raise FileNotFoundError(f"Input image not found: {IN_PATH}")

img = Image.open(IN_PATH).convert("RGBA")
width, height = img.size

# upscale factors to produce
factors = [2, 4]

for f in factors:
    new_size = (width * f, height * f)
    up = img.resize(new_size, resample=Image.LANCZOS)
    # apply a gentle unsharp mask to increase perceptual sharpness
    up = up.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    out_name = f"methodology_upscaled_{f}x.png"
    out_path = os.path.join(ROOT, "diagrams", out_name)
    up.save(out_path)
    print(f"Saved: {out_path}")

print("Done.")
