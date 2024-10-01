from ultralytics import YOLO  
import argparse
import os
import cv2  


class Object:
    def __init__(self, bbox: list[int], category: str):
        self.bbox = bbox
        self.category = category

    def __repr__(self):
        return f"Object(bbox={self.bbox}, category='{self.category}')"


class ObjectDetectionService:
    
    model = YOLO('yolov8n.pt')  

    @staticmethod
    def fun1(image_path: str) -> int:
        
        results = ObjectDetectionService.model(image_path) 
        categories = set([ObjectDetectionService.model.names[int(box.cls)] for box in results[0].boxes])
        return len(categories)

    @staticmethod
    def fun2(image_path: str) -> list[Object]:
        
        results = ObjectDetectionService.model(image_path)  
        detected_objects = []

        for box in results[0].boxes:
            bbox = [int(coord) for coord in box.xyxy[0].tolist()]  
            category = ObjectDetectionService.model.names[int(box.cls)]  
            detected_objects.append(Object(bbox, category))

        return detected_objects

def find_images_in_current_directory(directory: str = None) -> list[str]:
    
    directory = directory if directory is not None else os.getcwd()
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def main(image_paths: list[str]):
    
    for image_path in image_paths:
        print(f"\nProcessing image: {image_path}")
        num_categories = ObjectDetectionService.fun1(image_path)
        print(f"Number of unique categories: {num_categories}")

        detected_objects = ObjectDetectionService.fun2(image_path)
        print("Detected objects:")
        for obj in detected_objects:
            print(obj)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Object Detection Server for COCO Dataset")
    parser.add_argument('--file_to_image', type=str, help="Path to the image or directory of images")
    parser.add_argument('--image_path', type=str, help="Path to the image file")

    args = parser.parse_args()

    directory = args.file_to_image if args.file_to_image else None

    image_paths = find_images_in_current_directory(directory)

    if not image_paths and args.image_path:
        image_paths = [args.image_path]

    if not image_paths:
        print("No images found in the specified directory or no image path provided.")
    else:
        main(image_paths)
