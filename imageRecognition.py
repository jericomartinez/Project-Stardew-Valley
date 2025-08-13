from ultralytics import YOLO
import pyautogui

model = YOLO('best.pt')

screen_width, screen_height = pyautogui.size()

# Calculate center
screenx_center = screen_width // 2
screeny_center = screen_height // 2

decision = {
    "tree": False,
    "stick": False,
    "rock": False,
    "weed": False,
    "tree stump": False,
}

results = model(['test1.png'], conf=.80, save=True)  # return a list of Results objects
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()

for box, cls, conf in zip(boxes, classes, confidences):
    x1, y1, x2, y2 = box
    
    center_x = (x1+x2) / 2
    center_y = (y1+y2) / 2

    confidence = conf
    detected_class = cls
    name = names[int(cls)]
    
    if name == "tree":
        decision["tree"] = True
        # euclidean distance
        distance = ((center_x - screenx_center) ** 2 + (center_y - screeny_center) **2) **.5
        if "tree location" in decision:
            # Calculate if closer
            if distance < decision["tree distance"]:
                decision["tree location"] = (center_x, center_y)
                decision["tree distance"] = distance
        else:
            decision["tree location"] = (center_x, center_y)
            decision["tree distance"] = distance
    elif name == "stick":
        decision["stick"] = True
        # euclidean distance
        distance = ((center_x - screenx_center) ** 2 + (center_y - screeny_center) **2) **.5
        if "stick location" in decision:
            if distance < decision["stick distance"]:
                decision["stick location"] = (center_x, center_y)
                decision["stick distance"] = distance
        else:
            decision["stick location"] = (center_x, center_y)
            decision["stick distance"] = distance
    elif name == "rock":
        decision["rock"] = True
        # euclidean distance
        distance = ((center_x - screenx_center) ** 2 + (center_y - screeny_center) **2) **.5
        if "rock location" in decision:
            # Calculate if closer
            if distance < decision["rock distance"]:
                decision["rock location"] = (center_x, center_y)
                decision["rock distance"] = distance
        else:
            decision["rock location"] = (center_x, center_y)
            decision["rock distance"] = distance
    elif name == "weed":
        decision["weed"] = True
        # euclidean distance
        distance = ((center_x - screenx_center) ** 2 + (center_y - screeny_center) **2) **.5
        if "weed location" in decision:
            # Calculate if closer
            if distance < decision["weed distance"]:
                decision["weed location"] = (center_x, center_y)
                decision["weed distance"] = distance
        else:
            decision["weed location"] = (center_x, center_y)
            decision["weed distance"] = distance
    elif name == "tree stump":
        decision["tree stump"] = True
        # euclidean distance
        distance = ((center_x - screenx_center) ** 2 + (center_y - screeny_center) **2) **.5
        if "tree stump location" in decision:
            # Calculate if closer
            if distance < decision["tree stump distance"]:
                decision["tree stump location"] = (center_x, center_y)
                decision["tree stump distance"] = distance
        else:
            decision["weed location"] = (center_x, center_y)
            decision["weed distance"] = distance
    
print(decision)