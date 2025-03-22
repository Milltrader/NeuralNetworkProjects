import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

RARE_CLASSES = ['No Helmet', 'No Vest']

augmentations = A.Compose([
    A.HorizontalFlip(p=0.5), 
    A.RandomBrightnessContrast(p=0.2), 
    A.Rotate(limit=20, p=0.5), 
    A.MotionBlur(blur_limit=3, p=0.2),
    A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_image(image_path, label_path, AUG_IMAGES_DIR, AUG_LABELS_DIR):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    # load bounding boxes

    bboxes = []
    labels = []
    no_helmet = False
    no_vest = False

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                class_id = int(float(parts[0]))
                x_center, y_center, w, h = map(float, parts[1:])
                bboxes.append([x_center, y_center, w, h])
                labels.append(class_id)


                if class_id  == 1:
                    no_helmet = True
                elif class_id == 2:
                    no_vest = True  
    
    num_augments = 8 if no_helmet else (2 if no_vest else 1)   # Less augmentation for common classes

    # Apply augmentations

    for i in range(num_augments):
        augmented = augmentations(image=img, bboxes=bboxes, class_labels=labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']   
        aug_labels = augmented['class_labels']

        # save augmented image
        new_image_name = f"aug_{i}_{os.path.basename(image_path)}"
        new_image_path = os.path.join(AUG_IMAGES_DIR, new_image_name)
        cv2.imwrite(new_image_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        new_label_name = new_image_name.replace(".jpg", ".txt").replace(".png", ".txt")
        with open(os.path.join(AUG_LABELS_DIR, new_label_name), "w") as f:
            for bbox, label in zip(aug_bboxes, aug_labels):
                f.write(f"{label} {' '.join(map(str, bbox))}\n")