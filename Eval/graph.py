import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
models = ['LLAMA3.1-8B-Instruct', 'LLAMA3.2-3B-Instruct', 'LLAMA3.2-3B']
mmlu_kor_scores = [49.0010, 42.547, 33.7648]
mmlu_org_scores = [66.4067, 60.3060, 55.5904]

# --- Plotting ---

# Set up positions for the bars
x = np.arange(len(models))  # the label locations [0, 1, 2]
width = 0.35  # the width of the bars

# Create the figure and axes object
fig, ax = plt.subplots(figsize=(12, 7)) # Adjust figure size for better readability

# Plot the bars for each dataset
rects1 = ax.bar(x - width/2, mmlu_kor_scores, width, label='MMLU_KOR', color='skyblue')
rects2 = ax.bar(x + width/2, mmlu_org_scores, width, label='MMLU_ORG (Original)', color='lightcoral')

# Add labels, title, and ticks
ax.set_ylabel('MMLU Accuracy (%)') # Assuming the scores are accuracy percentages
ax.set_title('MMLU Benchmark Results Comparison by Model and Dataset')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right') # Rotate labels slightly if needed
ax.legend()

# Add value labels on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', # Format to 2 decimal places
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Add a grid for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(0, max(mmlu_org_scores) * 1.15) # Set y-limit slightly higher than max score

# Adjust layout to prevent labels overlapping
fig.tight_layout()

# Show the plot
plt.show()

# Optional: Save the plot
fig.savefig("mmlu_comparison_chart.png", dpi=300)
logger.info("Chart saved as mmlu_comparison_chart.png") # If using logging