from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from texturelab.config import EdgeConfig
from texturelab.edge_branch import detect_edges

image_path = Path(__file__).resolve().parent.parent / "Oval.png"

gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0

config_4 = EdgeConfig(connectivity=4)
result_4 = detect_edges(gray, config_4)

config_8 = EdgeConfig(connectivity=8)
result_8 = detect_edges(gray, config_8)

edges_4 = result_4["edges"]
edges_8 = result_8["edges"]

print("4-connectivity edge pixels:", np.count_nonzero(edges_4))
print("8-connectivity edge pixels:", np.count_nonzero(edges_8))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original grayscale image")

axes[1].imshow(edges_4, cmap="gray")
axes[1].set_title("Edges: connectivity = 4")

axes[2].imshow(edges_8, cmap="gray")
axes[2].set_title("Edges: connectivity = 8")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()