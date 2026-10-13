from neural_network import DIGIT_STROKES, STROKE_REGIONS


SEGMENT_PERCENT_DEFINITIONS = {
    "top": (0.00, 0.25, 0.25, 0.75),
    "middle": (0.38, 0.62, 0.25, 0.75),
    "bottom": (0.75, 1.00, 0.25, 0.75),
    "upper_left": (0.12, 0.50, 0.12, 0.38),
    "lower_left": (0.50, 0.88, 0.12, 0.38),
    "upper_right": (0.12, 0.50, 0.62, 0.88),
    "lower_right": (0.50, 0.88, 0.62, 0.88),
    "center_vertical": (0.12, 0.88, 0.38, 0.62),
}


def rectangle_region(size, top, bottom, left, right):
    row_start = max(0, round(size * top))
    row_end = min(size, round(size * bottom))
    col_start = max(0, round(size * left))
    col_end = min(size, round(size * right))

    return [
        (row, col)
        for row in range(row_start, row_end)
        for col in range(col_start, col_end)
    ]


def bounding_box(region):
    rows = [row for row, _ in region]
    cols = [col for _, col in region]
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def format_cells(region):
    return ", ".join(f"({row},{col})" for row, col in region)


def print_grid(region, size):
    region_cells = set(region)
    print("    cols: " + " ".join(str(col) for col in range(size)))
    for row in range(size):
        cells = ["#" if (row, col) in region_cells else "." for col in range(size)]
        print(f"    row {row}: " + " ".join(cells))


def print_task_regions():
    print("=== neural_network.py / STROKE_REGIONS, original 8x8 dataset ===")
    for name, region in STROKE_REGIONS.items():
        row_start, row_end, col_start, col_end = bounding_box(region)
        print()
        print(name)
        print(f"  rows: range({row_start}, {row_end})")
        print(f"  cols: range({col_start}, {col_end})")
        print(f"  pixels: {len(region)}")
        print(f"  cells: {format_cells(region)}")
        print_grid(region, size=8)


def print_segment_regions(size):
    print()
    print(f"=== compare_versions.py / make_segment_regions({size}) ===")
    for name, (top, bottom, left, right) in SEGMENT_PERCENT_DEFINITIONS.items():
        region = rectangle_region(size, top, bottom, left, right)
        row_start, row_end, col_start, col_end = bounding_box(region)
        print()
        print(name)
        print(f"  percent definition: top={top}, bottom={bottom}, left={left}, right={right}")
        print(f"  computed rows: range({row_start}, {row_end})")
        print(f"  computed cols: range({col_start}, {col_end})")
        print(f"  pixels: {len(region)}")
        print(f"  cells: {format_cells(region)}")
        print_grid(region, size=size)


def print_digit_strokes():
    print()
    print("=== DIGIT_STROKES, which regions each digit expects ===")
    for digit, strokes in DIGIT_STROKES.items():
        print(f"  {digit}: {', '.join(strokes)}")


def main():
    print_task_regions()
    print_segment_regions(size=16)
    print_digit_strokes()


if __name__ == "__main__":
    main()
