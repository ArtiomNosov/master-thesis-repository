from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source-original.png"
OUTPUT = HERE / "ai-configuration-ru-raster.png"

PAGE = (249, 250, 251)
WHITE = (255, 255, 255)
TEXT = (17, 24, 39)
BODY = (31, 41, 55)
MUTED = (107, 114, 128)
HINT = (156, 163, 175)
BLUE = (37, 99, 235)


def font_path(name: str) -> str:
    windows = Path("C:/Windows/Fonts") / name
    if windows.exists():
        return str(windows)
    return name


FONT_REGULAR = font_path("segoeui.ttf")
FONT_SEMIBOLD = font_path("seguisb.ttf")
FONT_BOLD = font_path("segoeuib.ttf")


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.truetype("DejaVuSans.ttf", size)


def fit_font(path: str, text: str, size: int, max_width: int, min_size: int = 10) -> ImageFont.FreeTypeFont:
    for current in range(size, min_size - 1, -1):
        candidate = font(path, current)
        if candidate.getlength(text) <= max_width:
            return candidate
    return font(path, min_size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font_obj: ImageFont.FreeTypeFont, fill=BODY) -> None:
    draw.text(xy, value, font=font_obj, fill=fill)


def segmented_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    strong: str,
    muted: str,
    strong_font: ImageFont.FreeTypeFont,
    muted_font: ImageFont.FreeTypeFont,
) -> None:
    x, y = xy
    text(draw, (x, y), strong, strong_font, TEXT)
    x += round(strong_font.getlength(strong))
    text(draw, (x, y), muted, muted_font, HINT)


def draw_save_icon(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    stroke = WHITE
    draw.rounded_rectangle((x, y, x + 12, y + 14), radius=1, outline=stroke, width=2)
    draw.line((x + 3, y, x + 3, y + 5, x + 9, y + 5), fill=stroke, width=2)
    draw.line((x + 3, y + 14, x + 3, y + 9, x + 9, y + 9, x + 9, y + 14), fill=stroke, width=2)


def main() -> None:
    img = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(img)

    title_font = font(FONT_BOLD, 18)
    subtitle_font = font(FONT_REGULAR, 14)
    section_title_font = font(FONT_BOLD, 16)
    section_subtitle_font = font(FONT_REGULAR, 14)
    label_font = font(FONT_SEMIBOLD, 14)
    input_font = font(FONT_REGULAR, 14)
    helper_font = font(FONT_REGULAR, 12)
    helper_small_font = fit_font(
        FONT_REGULAR,
        "Найдите цены на сайте провайдера. Расходы считаются локально — ничего не отправляется вовне.",
        12,
        625,
        11,
    )
    button_font = fit_font(FONT_BOLD, "Сохранить конфигурацию", 14, 178, 12)

    # Page heading.
    draw.rectangle((46, 27, 620, 72), fill=PAGE)
    text(draw, (46, 27), "Конфигурация ИИ", title_font, TEXT)
    text(draw, (46, 56), "Настройте провайдера ИИ и модель для оценки анализа кандидатов.", subtitle_font, MUTED)

    # Provider card header.
    draw.rectangle((122, 120, 650, 160), fill=WHITE)
    text(draw, (122, 122), "Провайдер и модель", section_title_font, TEXT)
    text(draw, (122, 146), "Выберите провайдера ИИ и модель.", section_subtitle_font, MUTED)

    # Provider card fields.
    draw.rectangle((68, 202, 230, 224), fill=WHITE)
    text(draw, (70, 206), "Провайдер", label_font, TEXT)

    draw.rectangle((82, 235, 360, 261), fill=WHITE)
    text(draw, (86, 240), "Микросервис bi-encoder", input_font, BODY)

    draw.rectangle((68, 286, 180, 310), fill=WHITE)
    text(draw, (70, 291), "Модель", label_font, TEXT)

    draw.rectangle((82, 318, 292, 344), fill=WHITE)
    text(draw, (86, 322), "cointegrated/rubert-tiny2", input_font, BODY)

    draw.rectangle((68, 352, 692, 377), fill=WHITE)
    text(
        draw,
        (70, 356),
        "Выберите предложенную модель или введите имя модели, которое поддерживает провайдер.",
        helper_font,
        HINT,
    )

    draw.rectangle((68, 393, 240, 414), fill=WHITE)
    text(draw, (70, 397), "Максимум токенов", label_font, TEXT)

    draw.rectangle((82, 424, 160, 450), fill=WHITE)
    text(draw, (86, 428), "4096", input_font, BODY)

    draw.rectangle((68, 460, 575, 485), fill=WHITE)
    text(draw, (70, 464), "Максимум токенов для результата анализа ИИ (256-32 768).", helper_font, HINT)

    # Wider localized button.
    draw.rectangle((69, 504, 310, 542), fill=WHITE)
    draw.rounded_rectangle((70, 505, 306, 541), radius=7, fill=BLUE)
    draw_save_icon(draw, 88, 516)
    text(draw, (110, 514), "Сохранить конфигурацию", button_font, WHITE)

    # Cost card header.
    draw.rectangle((122, 618, 700, 657), fill=WHITE)
    text(draw, (122, 620), "Учет затрат", section_title_font, TEXT)
    text(
        draw,
        (122, 644),
        "Задайте цены модели, чтобы отслеживать расходы на странице анализа ИИ.",
        section_subtitle_font,
        MUTED,
    )

    # Cost fields.
    draw.rectangle((68, 702, 355, 725), fill=WHITE)
    segmented_label(
        draw,
        (70, 706),
        "Цена ввода",
        " / 1 млн токенов",
        label_font,
        label_font,
    )

    draw.rectangle((388, 702, 697, 725), fill=WHITE)
    segmented_label(
        draw,
        (390, 706),
        "Цена вывода",
        " / 1 млн токенов",
        label_font,
        label_font,
    )

    draw.rectangle((82, 733, 170, 758), fill=WHITE)
    text(draw, (84, 737), "$", input_font, MUTED)
    text(draw, (112, 737), "0,15", input_font, BODY)

    draw.rectangle((402, 733, 490, 758), fill=WHITE)
    text(draw, (404, 737), "$", input_font, MUTED)
    text(draw, (432, 737), "0,6", input_font, BODY)

    draw.rectangle((68, 783, 707, 807), fill=WHITE)
    text(
        draw,
        (70, 787),
        "Найдите цены на сайте провайдера. Расходы считаются локально — ничего не отправляется вовне.",
        helper_small_font,
        HINT,
    )

    img.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
