from ultralytics import YOLO  
import argparse
import os
import cv2  
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


class Object:
    def __init__(self, bbox: list[int], category: str):
        self.bbox = bbox
        self.category = category

    def __repr__(self):
        return f"Object(bbox={self.bbox}, category='{self.category}')"


class ObjectDetectionService:
    
    model = YOLO('yolov8n.pt')  
    #print(model)
    @staticmethod
    def fun1(image_path: str) -> int:
        
        results = ObjectDetectionService.model(image_path, verbose=False) 
        categories = set([ObjectDetectionService.model.names[int(box.cls)] for box in results[0].boxes])
        return len(categories)

    @staticmethod
    def fun2(image_path: str) -> list[Object]:
        
        results = ObjectDetectionService.model(image_path, verbose=False)  
        detected_objects = []

        for box in results[0].boxes:
            bbox = [int(coord) for coord in box.xyxy[0].tolist()]  
            category = ObjectDetectionService.model.names[int(box.cls)]  
            detected_objects.append(Object(bbox, category))

        return detected_objects

def find_images_in_current_directory(directory: str | None = None) -> list[str]:
    
    directory = directory if directory is not None else os.getcwd()
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def load_coco_annotations(annotation_file: str):
    return COCO(annotation_file)

def show_image_with_annotations(image_path: str, detected_objects: list[Object], true_objects: list[Object], missed_items: list[Object]):
    image = cv2.imread(image_path)
    if len(missed_items) > 0:
        for obj in missed_items:
            bbox = obj.bbox
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)  # Red for missed items
            cv2.putText(image, f'Missed: {obj.category}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        for obj in detected_objects:
            bbox = obj.bbox
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)  # Blue for detected
            cv2.putText(image, obj.category, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for obj in true_objects:
            bbox = obj.bbox
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Green for ground truth
            cv2.putText(image, f'True: {obj.category}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def compare_detections_and_ground_truth(detected_objects: list[Object], true_objects: list[Object]):
    missed_items = []
    detected_categories = [obj.category for obj in detected_objects]

    for true_obj in true_objects:
        if true_obj.category not in detected_categories:
            missed_items.append(true_obj)
            print(f"Missed item: {true_obj.category}, BBox: {true_obj.bbox}")
    return missed_items

def main(image_paths: list[str], coco: COCO):
    for image_path in image_paths:
        print(f"\nProcessing image: {image_path}")
    
    
        num_categories = ObjectDetectionService.fun1(image_path)
        print(f"Number of unique categories: {num_categories}")
        detected_objects = ObjectDetectionService.fun2(image_path)
        print("Detected objects:")
        for obj in detected_objects:
            print(obj)

        image_id = int(os.path.splitext(os.path.basename(image_path))[0].split('_')[-1])  # Assuming ID is in the filename

        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)
        true_objects = [Object([int(coord) for coord in ann['bbox']], coco.loadCats(ann['category_id'])[0]['name']) for ann in annotations]
        
        missed_items = compare_detections_and_ground_truth(detected_objects, true_objects)
        print(f"Number of missed items: {len(missed_items)}")
        show_image_with_annotations(image_path, detected_objects, true_objects, missed_items)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Object Detection Server for COCO Dataset")
    parser.add_argument('--directory', type=str, help="Path to the image or directory of images")
    parser.add_argument('--image', type=str, help="Path to the image file")
    parser.add_argument('--annotations', type=str, required=True, help="Path to the COCO annotations JSON file")

    args = parser.parse_args()

    directory = args.directory if args.directory else None

    image_paths = find_images_in_current_directory(directory)

    if not image_paths and args.image:
        image_paths = [args.image]

    if not image_paths:
        print("No images found or no image path provided.")
    else:
        coco = load_coco_annotations(args.annotations)
        main(image_paths, coco)
