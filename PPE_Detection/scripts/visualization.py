import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_MAP = {
    0: 'Helmet',
    1: 'No Helmet',
    2: 'No Vest',
    3: 'Person',
    4: 'Vest'
}

def visualize_batch_on_grid(DATASET_PATH, split='train', num_images=10):
    images_dir = os.path.join(DATASET_PATH, split, 'images')
    labels_dir = os.path.join(DATASET_PATH, split, 'labels')

    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_images = min(num_images, len(all_images))
    sample_images = np.random.choice(all_images, num_images, replace=False)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, sample_image_name in enumerate(sample_images):
        sample_image_path = os.path.join(images_dir, sample_image_name)
        if not os.path.exists(sample_image_path):
            print(f"Image not found: {sample_image_name}")
            continue
        sample_image = cv2.imread(sample_image_path)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(labels_dir, os.path.splitext(sample_image_name)[0] + '.txt')
        if not os.path.exists(label_path):
            print(f"Label not found for {sample_image_name}")
            continue

        h, w, _ = sample_image.shape
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                class_id, x_center, y_center, width, height = map(float, parts)
                x1 = int((x_center - width / 2) * w)
                x2 = int((x_center + width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                y2 = int((y_center + height / 2) * h)
                label = CLASS_MAP.get(int(class_id), 'Unknown')
                colour = (0, 0, 255) if label == 'Person' else (255, 0, 0) if 'no ' in label.lower() else (0, 255, 0)
                cv2.rectangle(sample_image, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(sample_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        axes[idx].imshow(sample_image)
        axes[idx].set_title(f"{sample_image_name[:6]}")
        axes[idx].axis('off')
    for i in range(len(sample_images), 10):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()

def plot_class_distribution(class_distribution, title):
    class_names, class_counts = zip(*sorted(class_distribution.items(), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(5, 3))
    sns.barplot(x=class_counts, y=class_names, palette="viridis")
    plt.xlabel("Number of Images")
    plt.ylabel("Class")
    plt.title(title)
    for i, v in enumerate(class_counts):
        plt.text(v + 2, i, str(v), va='center')
    plt.show()
