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
    rounded(d, (4 * scale, 4 * scale, 124 * scale, 124 * scale), 28 * scale, PAPER, GREEN_DARK, max(2, int(8 * scale)))
    d.text(
        (64 * scale, 61 * scale),
        "U",
        fill=GREEN_DARK,
        font=font(max(12, int(82 * scale)), bold=True, serif=True),
        anchor="mm",
    )
    d.ellipse((74 * scale, 68 * scale, 100 * scale, 94 * scale), fill=BLUE)
    return img


def write_icons():
    for size in (16, 32, 48, 128):
      draw_icon(size).save(ICON_DIR / f"icon-{size}.png")


def text(draw, xy, value, fill=INK, size=24, bold=False, serif=False, anchor=None):
    draw.text(xy, value, fill=fill, font=font(size, bold=bold, serif=serif), anchor=anchor)


def listing_screenshot():
    img = Image.new("RGB", (1280, 800), PAPER)
    d = ImageDraw.Draw(img)
    text(d, (72, 72), "SumForU", MUTED, 28, bold=True)
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


def promo_tile(output_name="small-promo-440x280.png", caption_layout="two-line-center"):
    img = Image.new("RGB", (440, 280), GREEN_DARK)
    d = ImageDraw.Draw(img)
    d.ellipse((318, -44, 498, 136), fill="#296f55")
    d.ellipse((-52, 182, 84, 318), fill="#315f8c")
    rounded(d, (34, 58, 142, 166), 28, PAPER)
    icon = draw_icon(82).convert("RGBA")
    img.paste(icon, (47, 71), icon)
    text(d, (168, 68), "SumForU", "white", 30, bold=True)
    text(d, (168, 113), "Know if it fits.", "#fff7eb", 28, serif=True, bold=True)
    if caption_layout == "one-line-center":
        text(d, (220, 211), "Personal buying summaries from product pages.", "#d9e6dc", 17, anchor="mm")
    elif caption_layout == "two-line-center":
        text(d, (220, 199), "Personal buying summaries", "#d9e6dc", 19, anchor="mm")
        text(d, (220, 226), "from real product pages.", "#d9e6dc", 19, anchor="mm")
    else:
        text(d, (44, 199), "Personal buying summaries", "#d9e6dc", 19)
        text(d, (44, 226), "from real product pages.", "#d9e6dc", 19)
    img.save(STORE_DIR / output_name)


def marquee_tile():
    img = Image.new("RGB", (1400, 560), PAPER)
    d = ImageDraw.Draw(img)
    d.ellipse((1110, -180, 1510, 220), fill="#e7efe7")
    d.ellipse((-130, 330, 250, 710), fill="#e8eef4")
    rounded(d, (80, 86, 378, 384), 58, GREEN_DARK)
    icon = draw_icon(214).convert("RGBA")
    img.paste(icon, (122, 128), icon)
    text(d, (455, 122), "SumForU", INK, 68, bold=True, serif=True)
    text(d, (455, 216), "A buying fit check for the page", MUTED, 34)
    text(d, (455, 258), "you are already reading.", MUTED, 34)

    rounded(d, (455, 340, 1040, 438), 24, PANEL, LINE, 2)
    rounded(d, (487, 370, 589, 410), 14, GREEN)
    text(d, (538, 378), "Good fit", "white", 19, bold=True, anchor="ma")
    text(d, (620, 360), "Review signals, price, specs, and tradeoffs", INK, 24, bold=True)
    text(d, (620, 396), "compressed into one personal recommendation.", MUTED, 21)

    rounded(d, (1100, 338, 1220, 438), 24, BLUE)
    text(d, (1160, 354), "8", "white", 50, bold=True, anchor="ma")
    text(d, (1192, 376), "/10", "white", 24, bold=True, anchor="ma")
    img.save(STORE_DIR / "marquee-promo-1400x560.png")


def large_promo_lens_left():
    img = Image.new("RGB", (1400, 560), GREEN_DARK)
    d = ImageDraw.Draw(img)
    d.ellipse((1120, -210, 1530, 200), fill="#296f55")
    d.ellipse((-160, 360, 240, 760), fill="#315f8c")

    rounded(d, (96, 96, 420, 420), 64, PAPER)
    icon = draw_icon(230).convert("RGBA")
    img.paste(icon, (143, 143), icon)

    text(d, (510, 116), "SumForU", "white", 66, bold=True, serif=True)
    text(d, (510, 220), "Shop with your own lens.", "#fff7eb", 50, bold=True, serif=True)
    text(d, (510, 330), "See what matters to you across reviews,", "#d9e6dc", 30)
    text(d, (510, 372), "price, and product details.", "#d9e6dc", 30)

    rounded(d, (510, 450, 790, 500), 16, "#296f55")
    text(d, (650, 463), "Personal buying summary", "white", 22, bold=True, anchor="ma")
    img.save(STORE_DIR / "large-promo-option-a-lens-left.png")


def large_promo_centered(output_name="large-promo-option-b-centered.png"):
    img = Image.new("RGB", (1400, 560), PAPER)
    d = ImageDraw.Draw(img)
    d.ellipse((990, -250, 1510, 270), fill="#e7efe7")
    d.ellipse((-180, 350, 280, 810), fill="#e8eef4")

    rounded(d, (556, 56, 844, 344), 58, GREEN_DARK)
    icon = draw_icon(198).convert("RGBA")
    img.paste(icon, (601, 101), icon)
    text(d, (700, 404), "SumForU", INK, 58, bold=True, serif=True, anchor="mm")
    text(d, (700, 470), "Shop with your own lens.", INK, 38, bold=True, serif=True, anchor="mm")
    text(d, (700, 522), "See what matters to you across reviews, price, and product details.", MUTED, 25, anchor="mm")
    img.save(STORE_DIR / output_name)


def large_promo_product_cards():
    img = Image.new("RGB", (1400, 560), PAPER)
    d = ImageDraw.Draw(img)
    d.ellipse((1130, -160, 1510, 220), fill="#e7efe7")
    d.ellipse((-130, 330, 260, 720), fill="#e8eef4")

    icon = draw_icon(126).convert("RGBA")
    img.paste(icon, (92, 92), icon)
    text(d, (250, 106), "SumForU", INK, 62, bold=True, serif=True)
    text(d, (250, 200), "Shop with your own lens.", INK, 44, bold=True, serif=True)
    text(d, (250, 292), "See what matters to you across", MUTED, 28)
    text(d, (250, 332), "reviews, price, and product details.", MUTED, 28)

    card_x = 910
    for idx, (label, value, color) in enumerate(
        [
            ("Reviews", "real user signal", GREEN),
            ("Price", "value context", BLUE),
            ("Details", "specs and tradeoffs", CORAL),
        ]
    ):
        y = 108 + idx * 104
        rounded(d, (card_x, y, 1250, y + 76), 22, PANEL, LINE, 2)
        d.ellipse((card_x + 24, y + 24, card_x + 52, y + 52), fill=color)
        text(d, (card_x + 70, y + 18), label, INK, 22, bold=True)
        text(d, (card_x + 70, y + 46), value, MUTED, 18)

    rounded(d, (930, 438, 1230, 506), 22, GREEN_DARK)
    text(d, (1080, 455), "Personal recommendation", "white", 24, bold=True, anchor="ma")
    img.save(STORE_DIR / "large-promo-option-c-product-cards.png")


def large_promo_candidates():
    large_promo_lens_left()
    large_promo_centered()
    large_promo_product_cards()


def final_marquee_tile():
    large_promo_centered("marquee-promo-1400x560.png")


def main():
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    write_icons()
    listing_screenshot()
    promo_tile()
    final_marquee_tile()
    large_promo_candidates()


if __name__ == "__main__":
    main()
