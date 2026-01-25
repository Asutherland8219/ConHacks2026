import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

from captcha_image_analysis import GridAnalyzer

def visualize_grid_analysis(image_path: str, output_path: str):
    """
    Runs grid analysis and saves a debug image with cell annotations.
    """
    analyzer = GridAnalyzer()
    analysis = analyzer.analyze_grid(image_path, save_analysis=True)

    if not analysis:
        print("Grid analysis failed.")
        return

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    try:
        # Use a truetype font if available, otherwise default
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    print("\n--- Grid Analysis Results ---")
    print(f"Grid size: {analysis['grid_size']}")
    print(f"Image cells: {analysis['image_cells']}")
    print(f"Color cells: {analysis['color_cells']}")
    print("-----------------------------\\n")

    for cell in analysis["cells"]:
        idx = cell["index"]
        bbox = cell["bbox"]
        cell_type = cell["type"]
        
        # Define a color for the bounding box
        outline_color = "red" if cell_type == "color" else "green"
        
        # Draw bounding box
        draw.rectangle(
            (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
            outline=outline_color,
            width=3,
        )

        # Prepare text to display
        label = f"#{idx}\n{cell_type}"
        
        # Add color info for color cells
        if cell_type == "color":
            color = cell.get('color', (0,0,0))
            label += f"\n{color}"

        # Draw text label
        text_position = (bbox[0] + 5, bbox[1] + 5)
        draw.text(text_position, label, fill=outline_color, font=font)

    img.save(output_path)
    print(f"Debug image saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 debug_grid.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "debug_grid.png"

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    visualize_grid_analysis(image_path, output_path)
