from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DATASET_IMAGE_DIR = Path("data/Puzzle/images")
OUTPUT_IMAGE_DIR = Path("outputs/2_puzzle")
COMPARISON_PATH = Path("outputs/2_puzzle_comparison.png")


def main() -> None:
    """Save a side-by-side comparison sheet for Part 2 puzzle outputs."""
    matching_image_pairs = []
    for dataset_image_path in sorted(DATASET_IMAGE_DIR.glob("*.png")):
        image_index = dataset_image_path.stem
        output_image_path = OUTPUT_IMAGE_DIR / f"view_{image_index}.png"
        if output_image_path.exists():
            matching_image_pairs.append((image_index, dataset_image_path, output_image_path))

    if not matching_image_pairs:
        raise RuntimeError("No matching dataset/output image pairs found.")

    font = ImageFont.load_default()
    thumbnail_size = 160
    label_height = 28
    pair_gap = 10
    cell_padding = 14
    pairs_per_row = 4
    cell_width = thumbnail_size * 2 + pair_gap + cell_padding * 2
    cell_height = thumbnail_size + label_height + cell_padding * 2
    row_count = (len(matching_image_pairs) + pairs_per_row - 1) // pairs_per_row

    comparison_sheet = Image.new("RGB", (pairs_per_row * cell_width, row_count * cell_height), "white")
    draw = ImageDraw.Draw(comparison_sheet)

    for pair_index, (image_index, dataset_image_path, output_image_path) in enumerate(matching_image_pairs):
        row = pair_index // pairs_per_row
        column = pair_index % pairs_per_row
        cell_x = column * cell_width + cell_padding
        cell_y = row * cell_height + cell_padding

        dataset_image = Image.open(dataset_image_path).convert("RGB").resize(
            (thumbnail_size, thumbnail_size),
            Image.Resampling.NEAREST,
        )
        output_image = Image.open(output_image_path).convert("RGB").resize(
            (thumbnail_size, thumbnail_size),
            Image.Resampling.NEAREST,
        )

        comparison_sheet.paste(dataset_image, (cell_x, cell_y + label_height))
        comparison_sheet.paste(output_image, (cell_x + thumbnail_size + pair_gap, cell_y + label_height))

        draw.text((cell_x, cell_y), f"{image_index}: dataset", fill="black", font=font)
        draw.text((cell_x + thumbnail_size + pair_gap, cell_y), "output", fill="black", font=font)

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_sheet.save(COMPARISON_PATH)
    print(f"Saved {COMPARISON_PATH} with {len(matching_image_pairs)} pairs")


if __name__ == "__main__":
    main()
