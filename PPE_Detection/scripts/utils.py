import os

def get_class_distribution(DATASET_PATH, split='train', CLASS_MAP=None):
    labels_dir = os.path.join(DATASET_PATH, split, 'labels')
    # Default CLASS_MAP if not provided
    if CLASS_MAP is None:
        CLASS_MAP = {0: 'Helmet', 1: 'No Helmet', 2: 'No Vest', 3: 'Person', 4: 'Vest'}
    class_distribution = {class_name: 0 for class_name in CLASS_MAP.values()}
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                class_id = int(float(parts[0]))
                class_name = CLASS_MAP.get(class_id, 'Unknown')
                class_distribution[class_name] += 1
    return class_distribution
