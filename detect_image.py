from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def get_class_name(index_value):
    file_path = "classes.txt"  # Replace with the actual path to your coco_classes.txt file
    
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            if 0 <= index_value < len(lines):
                nth_line = lines[index_value].strip()  # Read the n-th line and remove leading/trailing whitespace
                return nth_line
                print(f"Line {n+1}: {nth_line}")
            else:
                print(f"Line {n+1} does not exist in the file.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Load a pretrained YOLOv8n model
model = YOLO('ultralytics/yolov8n.pt')

# Reading Imag
image = Image.open('test/car.jpeg')
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

results = model('car.jpeg')  # results list, use save_txt = True to save results
predicted_classes = results[0].boxes.cls
objects_coordinates = results[0].boxes.xyxy
count = len(predicted_classes)

for i in range(count):
    class_index = int(predicted_classes[i].item())
    class_name = get_class_name(class_index)
    coordinates = objects_coordinates[i]
    print(f"{i}/{count} {class_name}, {coordinates}")

    x1, y1, x2, y2 = coordinates
    # Draw a rectangle around the object
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    # Add text for the class name
    draw.text((x1, y1 - 10), class_name, fill="red", font=font)

image.show()

