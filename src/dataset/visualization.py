import matplotlib.pyplot as plt


def visualize_width_height(w, h):
    """
    Visualizes width and height distribution in a dataset

    Args:
        w (list): Widths
        h (list): Heights

    Returns:
        Plots W vs H distribution, returns nothing
    """
    plt.scatter(w, h, alpha=0.3, color='blue', edgecolors='none', s=10)

    plt.title('Bounding Box Width vs Height Distribution')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
