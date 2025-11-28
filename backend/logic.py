def decide_action(detections):
    vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
    close_vehicles = []

    for det in detections:
        if det["cls"] in vehicle_classes:
            x1, y1, x2, y2 = det["bbox"]
            box_height = y2 - y1

            if box_height > 250:  # tuned for closeness
                close_vehicles.append(det)

    if close_vehicles:
        return "A vehicle is very close. Do NOT cross."

    people = [d for d in detections if d["cls"] == "person"]
    if people:
        return "A person is in front of you. Move slightly left or right."

    if len(detections) == 0:
        return "I do not see anything ahead."

    return "The path seems clear. You can proceed cautiously."
