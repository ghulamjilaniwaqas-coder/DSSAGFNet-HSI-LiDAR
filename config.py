# Configuration
PATCH_SIZE = 11
HALF_PATCH = PATCH_SIZE // 2  # Added for preprocessing

NUM_RUNS = 3
EPOCHS = 1
CLASS_INFO = [
    (1, "Healthy grass", 198, 1053),
    (2, "Stressed grass", 190, 1064),
    (3, "Synthetic grass", 192, 505),
    (4, "Trees", 188, 1058),
    (5, "Soil", 186, 1056),
    (6, "Water", 182, 141),
    (7, "Residential", 196, 1072),
    (8, "Commercial", 191, 1053),
    (9, "Road", 193, 1059),
    (10, "Highway", 191, 1036),
    (11, "Railway", 181, 1054),
    (12, "Parking lot 1", 192, 1041),
    (13, "Parking lot 2", 184, 285),
    (14, "Tennis court", 181, 247),
    (15, "Running track", 187, 473)
]
NUM_CLASSES = len(CLASS_INFO)