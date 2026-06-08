import matplotlib.pyplot as plt

classes = ["Back_Panel", "Top_Panel", "Bottom_Part", "Front_Panel", "Side_Panel", "Extra_Areas"]
counts = [197, 210, 213, 217, 434, 1608]
total = sum(counts)

fig, ax = plt.subplots(figsize=(10, 6))

# Highlight minority vs majority
colors = ['#ff9999', '#ff9999', '#ff9999', '#ff9999', '#66b3ff', '#99ff99']
bars = ax.bar(classes, counts, color=colors)

ax.set_title('Class Distribution (Structural Packaging)', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Annotations', fontsize=12)
ax.set_xlabel('YOLO Class', fontsize=12)

# Add percentage and count labels
for bar in bars:
    height = bar.get_height()
    percentage = (height / total) * 100
    ax.annotate(f'{height}\n({percentage:.1f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#99ff99', label='Majority Class (>50%)'),
    Patch(facecolor='#66b3ff', label='Moderate Class (~15%)'),
    Patch(facecolor='#ff9999', label='Minority Classes (<8%)')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig("class_distribution_histogram.png", dpi=300)
print("Saved class_distribution_histogram.png")
