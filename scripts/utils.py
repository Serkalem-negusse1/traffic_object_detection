#Utility functions to visualize images with bounding boxes (important for verifying dataset and predictions).
import cv2
import matplotlib.pyplot as plt

def draw_bboxes(image, bboxes, labels=None, colors=None):
    """
    Draw bounding boxes on the image.

    Args:
        image: np.array image (H x W x 3)
        bboxes: list of [xmin, ymin, xmax, ymax]
        labels: list of strings for each bbox (optional)
        colors: list of colors for each bbox (optional)
    """
    img = image.copy()
    for i, box in enumerate(bboxes):
        color = colors[i] if colors else (0, 255, 0)
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        if labels:
            cv2.putText(img, labels[i], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
    return img

def show_image(img, figsize=(10, 10)):
    """Helper function to display image in notebook or script"""
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
