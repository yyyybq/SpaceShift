"""Question templates for video-based QA on trajectory videos.

Three QA categories by trajectory pattern:
  - Linear  (approach, passby):  object size only (individual dimensions)
  - Circular (around, spherical): object size + distance to camera
  - Rotation (rotation):         object size + object-to-object distance
"""

DIMENSION_DEFINITION = (
    "Length and width are the two dimensions that define the 'base' of the "
    "object. Of these two base dimensions, let length be the longer and "
    "width be the shorter."
)

OBJECT_SIZE_TEMPLATE = (
    "What is the estimated {dimension} of the {object} in this video in meters? "
    + DIMENSION_DEFINITION
)

OBJECT_SIZE_COMPARISON_RELATIVE_TEMPLATE = (
    "What is the ratio of the {dimension} of {object1} to the "
    "{dimension} of {object2}? " + DIMENSION_DEFINITION
)

OBJECT_SIZE_COMPARISON_ABSOLUTE_TEMPLATE = (
    "If {object2} has a {dimension} of {obj2_dimension} meters, "
    "what is the {dimension} of {object1}? " + DIMENSION_DEFINITION
)

OBJECT_DISTANCE_TO_CAMERA_TEMPLATE = (
    "What is the distance from the camera to the {object} in meters?"
)

OBJECT_PAIR_DISTANCE_TEMPLATE = (
    "What is the absolute distance between {object1} and {object2} "
    "in meters?"
)

OBJECT_PAIR_DISTANCE_W_SIZE_TEMPLATE = (
    "What is the absolute distance between {object1} and {object2} "
    "in meters, given that the {dimension} of {object1} is "
    "{obj1_dimension} meter(s)?"
)

OBJECT_DISTANCE_COMPARISON_RELATIVE_TEMPLATE = (
    "What is the ratio of the distance between {objectA} and {objectB} "
    "to the distance between {objectX} and {objectY}?"
)

DISTANCE_POST_PROMPT = (
    "Please assume the distance is measured from the approximated center "
    "of each object, and briefly explain the method used to determine "
    "the center."
)

NA_POST_PROMPT = (
    "The answer should be a single NUMBER given to one decimal place."
)

POST_PROMPT = (
    "[Output]\n"
    "You have to end your response with the answer formatted in a "
    "dictionary: {{'answer': <answer>}}. For example, "
    "{{'answer': 'Z'}} or {{'answer': 0}} or "
    "{{'answer': [0, 0, 0]}}, depending on the question."
)

PATTERN_QA_CATEGORY: dict[str, str] = {
    "approach": "linear",
    "passby": "linear",
    "around": "circular",
    "spherical": "circular",
    "rotation": "rotation",
}
