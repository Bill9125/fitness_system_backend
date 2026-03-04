# Skeleton connections for pose estimation drawing
# Format: (start_index, end_index)
DEADLIFT_SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),  # Right arm
    (5, 7),
    (5, 6),
    (7, 9),
    (6, 8),  # Left arm
    (6, 12),
    (12, 14),
    (14, 16),  # Right leg
    (5, 11),
    (11, 13),
    (13, 15)  # Left leg
]

BENCHPRESS_TOP_SKELETON_CONNECTIONS = [(0,1),(0,2),(1,3),(2,3),(4,6),(5,7),(0,4),(1,5)]
BENCHPRESS_REAR_SKELETON_CONNECTIONS = [(0,1),(0,2),(2,4),(1,3),(3,5)]