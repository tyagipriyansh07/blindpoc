

def decide_action(detections):
    """
    Smarter rule-based logic for blind navigation.

    Fixes included:
    - Only consider NEAR objects (large + low on screen)
    - Low-confidence 'person' → treat as UNKNOWN obstacle
    - Very thin objects → NOT a person (bottle/pole fix)
    - Groups objects (people / vehicles / unknown)
    """

    if not detections:
        return "No clear detections. Path seems clear, proceed carefully."

    # --- TUNING PARAMETERS (for mobile camera) ---
    NEAR_HEIGHT_MIN = 0.22     # object height must be at least 22% of frame
    NEAR_CY_MIN = 0.55         # object center must be in lower half
    MIN_PERSON_CONF = 0.65     # person must have ≥ 65% confidence
    MIN_PERSON_WIDTH = 0.08    # if width < 8% of frame → NOT a person
    # ------------------------------------------------

    vehicle_classes = {"car", "bus", "truck", "motorcycle", "bicycle"}
    person_cls = "person"

    near_people = 0
    near_vehicles = 0
    near_other_obstacles = 0

    for det in detections:
        cls = det["cls"]
        conf = det["conf"]
        cy = det["cy"]
        rel_h = det["rel_h"]
        rel_w = det["rel_w"]

        # Ignore far or irrelevant detections
        if rel_h < NEAR_HEIGHT_MIN or cy < NEAR_CY_MIN:
            continue

        # --- FIX 1: Low-confidence PERSON → unknown obstacle ---
        if cls == person_cls and conf < MIN_PERSON_CONF:
            near_other_obstacles += 1
            continue

        # --- FIX 2: Very thin shapes are NOT real people ---
        if cls == person_cls and rel_w < MIN_PERSON_WIDTH:
            near_other_obstacles += 1
            continue

        # --- REAL PERSON ---
        if cls == person_cls:
            near_people += 1
            continue

        # --- VEHICLE ---
        if cls in vehicle_classes:
            near_vehicles += 1
            continue

        # --- UNKNOWN OBJECT ---
        near_other_obstacles += 1

    # If nothing meaningful is near
    if near_people == near_vehicles == near_other_obstacles == 0:
        return "No close obstacles. Path ahead looks clear."

    # Build message parts
    messages = []

    if near_people > 0:
        msg = "one person" if near_people == 1 else f"{near_people} people"
        messages.append(f"{msg} close ahead")

    if near_vehicles > 0:
        msg = "one vehicle" if near_vehicles == 1 else f"{near_vehicles} vehicles"
        messages.append(f"{msg} ahead")

    if near_other_obstacles > 0:
        msg = "one object" if near_other_obstacles == 1 else f"{near_other_obstacles} objects"
        messages.append(f"{msg} on your path")

    return ", ".join(messages) + ". Move carefully."
