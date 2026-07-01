from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "extension" / "assets" / "icons"
STORE_DIR = ROOT / "store-assets"

INK = "#1e2428"
MUTED = "#687076"
PAPER = "#f6f3ec"
PANEL = "#ffffff"
LINE = "#d9d1c2"
GREEN = "#296f55"
GREEN_DARK = "#174d3a"
BLUE = "#315f8c"
CORAL = "#bf5a42"


def font(size, bold=False, serif=False):
    candidates = []
    if serif:
        candidates = [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_icon(size):
    scale = size / 128
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rounded(d, (4 * scale, 4 * scale, 124 * scale, 124 * scale), 24 * scale, GREEN_DARK)
    rounded(d, (16 * scale, 18 * scale, 112 * scale, 110 * scale), 16 * scale, PAPER)
    d.line((34 * scale, 45 * scale, 82 * scale, 45 * scale), fill=LINE, width=max(1, int(5 * scale)))
    d.line((34 * scale, 63 * scale, 96 * scale, 63 * scale), fill=LINE, width=max(1, int(5 * scale)))
    d.line((34 * scale, 81 * scale, 74 * scale, 81 * scale), fill=LINE, width=max(1, int(5 * scale)))
    d.ellipse((78 * scale, 70 * scale, 106 * scale, 98 * scale), fill=BLUE)
    d.text((88 * scale, 74 * scale), "Y", fill="white", font=font(max(12, int(20 * scale)), bold=True))
    return img


def write_icons():
    for size in (16, 32, 48, 128):
      draw_icon(size).save(ICON_DIR / f"icon-{size}.png")


def text(draw, xy, value, fill=INK, size=24, bold=False, serif=False, anchor=None):
    draw.text(xy, value, fill=fill, font=font(size, bold=bold, serif=serif), anchor=anchor)


def listing_screenshot():
    img = Image.new("RGB", (1280, 800), PAPER)
    d = ImageDraw.Draw(img)
    text(d, (72, 72), "Sum for You", MUTED, 28, bold=True)
    text(d, (72, 112), "A buying fit check", INK, 54, bold=True, serif=True)
    text(d, (72, 184), "for the page you're already reading", INK, 34, bold=True, serif=True)
    text(d, (72, 235), "Personalized summaries from reviews, price, and product details.", MUTED, 26)

    rounded(d, (70, 310, 740, 704), 24, PANEL, LINE, 2)
    text(d, (110, 355), "Product page", MUTED, 20, bold=True)
    text(d, (110, 395), "Apple AirPods Pro 2", INK, 34, bold=True, serif=True)
    for y, width in [(465, 470), (512, 540), (559, 430), (606, 500)]:
        rounded(d, (110, y, 110 + width, y + 18), 8, "#e8e0d3")
    rounded(d, (110, 655, 260, 692), 10, GREEN, None)
    text(d, (185, 663), "Reviews", "white", 18, bold=True, anchor="ma")

    rounded(d, (790, 128, 1192, 690), 24, "#fffaf2", LINE, 2)
    text(d, (832, 175), "RECOMMENDATION", MUTED, 18, bold=True)
    rounded(d, (1065, 162, 1145, 222), 14, BLUE)
    text(d, (1105, 174), "8", "white", 34, bold=True, anchor="ma")
    text(d, (1127, 188), "/10", "white", 18, bold=True, anchor="ma")
    text(d, (832, 215), "Good fit", INK, 34, bold=True, serif=True)
    text(d, (832, 280), "Strong ANC, easy Apple setup, and a", INK, 24, serif=True)
    text(d, (832, 314), "high review signal make this a good fit.", INK, 24, serif=True)
    text(d, (832, 390), "Strengths", INK, 22, bold=True)
    for idx, line in enumerate(["Comfortable for long wear", "Seamless iPhone/Mac pairing", "Very high review score"]):
        y = 435 + idx * 45
        d.ellipse((835, y + 8, 847, y + 20), fill=GREEN)
        text(d, (862, y), line, INK, 21)
    text(d, (832, 590), "Concerns", INK, 22, bold=True)
    d.ellipse((835, 635, 847, 647), fill=CORAL)
    text(d, (862, 627), "Premium price", INK, 21)
    img.save(STORE_DIR / "screenshot-1280x800.png")


def promo_tile():
    img = Image.new("RGB", (440, 280), GREEN_DARK)
    d = ImageDraw.Draw(img)
    rounded(d, (28, 34, 170, 176), 28, PAPER)
    icon = draw_icon(96).convert("RGBA")
    img.paste(icon, (51, 57), icon)
    text(d, (200, 58), "Sum for You", "white", 28, bold=True)
    text(d, (200, 102), "Know if it fits", "#e7efe7", 24, serif=True, bold=True)
    text(d, (200, 150), "Persona-aware buying", "#d9e6dc", 18)
    text(d, (200, 178), "summaries for product pages.", "#d9e6dc", 18)
    img.save(STORE_DIR / "small-promo-440x280.png")


def marquee_tile():
    img = Image.new("RGB", (1400, 560), PAPER)
    d = ImageDraw.Draw(img)
    rounded(d, (82, 82, 380, 380), 52, GREEN_DARK)
    icon = draw_icon(210).convert("RGBA")
    img.paste(icon, (126, 126), icon)
    text(d, (455, 135), "Sum for You", INK, 62, bold=True, serif=True)
    text(d, (455, 225), "Turn a product page into a short buying fit check.", MUTED, 34)
    rounded(d, (455, 315, 680, 375), 18, GREEN)
    text(d, (568, 331), "Generate", "white", 28, bold=True, anchor="ma")
    text(d, (455, 428), "Built for reviews, tradeoffs, and personal shopping context.", MUTED, 26)
    img.save(STORE_DIR / "marquee-promo-1400x560.png")


def main():
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    write_icons()
    listing_screenshot()
    promo_tile()
    marquee_tile()


if __name__ == "__main__":
    main()
