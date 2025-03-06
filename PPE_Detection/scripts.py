import os


def clean_labels(split, dataset_root):
    labels_dir = os.path.join(dataset_root, 'labels', split)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        
        label_path = os.path.join(labels_dir, label_file)

        cleaned_lines = []
        with open(label_path, 'r') as file:
            for line in file:
                class_id = int(line.split()[0])
                if class_id != 5:  # Skip class 5 (None)
                    cleaned_lines.append(line)

        # Overwrite the label file with cleaned content
        with open(label_path, 'w') as file:
            file.writelines(cleaned_lines)

        print(f"Cleaned {label_file}: removed {len(cleaned_lines)} lines (kept only non-None)")

# 