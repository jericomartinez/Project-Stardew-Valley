import threading
import pyautogui
from PIL import Image
from ultralytics import YOLO
import time

def run_bot(decision):
    # Performs actions based on detected objects
    if "tree location" in decision:
        pyautogui.press('1')
        pyautogui.moveTo(decision["tree location"])
        pyautogui.rightClick()
        print("Going to tree")
        time.sleep(6)
        pyautogui.mouseDown(decision["tree location"])
        print("Chopping tree down")
    elif "stick location" in decision:
        pyautogui.press('1')
        pyautogui.rightClick(decision["stick location"])
        time.sleep(6)
        pyautogui.mouseDown(decision["stick location"])
    elif "rock location" in decision:
        pyautogui.press('4')
        pyautogui.rightClick(decision["rock location"])
        time.sleep(6)
        pyautogui.mouseDown(decision["rock location"])
    elif "weed location" in decision:
        pyautogui.press('5')
        pyautogui.rightClick(decision["weed location"])
        time.sleep(6)
        pyautogui.mouseDown(decision["weed location"])
    elif "tree stump location" in decision:
        pyautogui.press('1')
        pyautogui.rightClick(decision["tree stump location"])
        time.sleep(6)
        pyautogui.mouseDown(decision["tree stump location"])


screen_width, screen_height = pyautogui.size()

# Thread function to take screenshots and detect objects
def take_screenshot(stop_event, model):
    screenx_center = screen_width // 2
    screeny_center = screen_height // 2

    pyautogui.FAILSAFE = False

    while not stop_event.is_set():

        decision = {
            "tree": False,
            "stick": False,
            "rock": False,
            "weed": False,
            "tree stump": False,
        }

        # Take screenshot
        screenshot = pyautogui.screenshot()
        # screenshot = Image.frombytes('RGB', screenshot.size, screenshot.tobytes())

        results = model([screenshot], conf=0.7)
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
                    
        run_bot(decision)

# Main function
def main():
    print("Starting bot. Press Enter to quit.")
    model = YOLO('best.pt')
    stop_event = threading.Event()

    screenshot_thread = threading.Thread(target=take_screenshot, args=(stop_event, model))
    screenshot_thread.start()

    # Wait for user to press Enter to quit
    input()
    stop_event.set()
    screenshot_thread.join()
    print("Bot stopped.")

if __name__ == "__main__":
    main()
