"""
Question Templates for Multiview QA Generation

Templates adapted from spatial-scene-variations/src/question_generation/question_templates.py
with multi-view support.
"""

# ==============================================================================
# SIZE QUESTIONS
# ==============================================================================

OBJECT_SIZE_TEMPLATE = """
What is the estimated length, width, and height of the {object} in this scene in meters? 
Length and width are the two dimensions that define the 'base' of the object. Of these two base dimensions, let length be the longer and width be the shorter.
The answer should be in the format: [length, width, height].
""".strip()

OBJECT_SIZE_COMPARISON_RELATIVE_TEMPLATE = """
What is the ratio of the {dimension} of {object1} to the {dimension} of {object2}? 
Length and width are the two dimensions that define the 'base' of the object. Of these two base dimensions, let length be the longer and width be the shorter.
""".strip()

OBJECT_SIZE_COMPARISON_ABSOLUTE_TEMPLATE = """
If {object2} has a {dimension} of {obj2_dimension} in meters, what is the {dimension} of {object1}? 
Length and width are the two dimensions that define the 'base' of the object. Of these two base dimensions, let length be the longer and width be the shorter.
""".strip()

# ==============================================================================
# DISTANCE QUESTIONS
# ==============================================================================

OBJECT_PAIR_DISTANCE_CENTER_TEMPLATE = """
What is the absolute distance between {object1} and {object2} in meters? 
""".strip()

OBJECT_COMPARISON_ABSOLUTE_DISTANCE_TEMPLATE = """
If the distance between {objectX} and {objectY} is {distance} meter(s), what is the distance between {objectA} and {objectB} in meters?
""".strip()

OBJECT_COMPARISON_RELATIVE_DISTANCE_TEMPLATE = """
What is the ratio of the distance between {objectA} and {objectB} to the distance between {objectX} and {objectY}?
""".strip()

OBJECT_PAIR_DISTANCE_CENTER_W_SIZE_TEMPLATE = """
What is the absolute distance between {object1} and {object2} in meters, given that the {dimension} of {object1} is {obj1_dimension} meter(s)?
""".strip()

OBJECT_DISTANCE_TO_CAMERA_TEMPLATE = """
What is the estimated distance of the {object1} from the camera in meters?
""".strip()

# ==============================================================================
# VECTOR QUESTIONS
# ==============================================================================

OBJECT_PAIR_DISTANCE_VECTOR_TEMPLATE = """
Using the agent's local coordinate system, provide the vector from the center of {objectA} to the center of {objectB} in meters.

The coordinate system is defined as follows:
* **Origin:** By definition, the agent is at the origin ([0,0,0]).
* **X-axis:** Left-to-right (positive X is to the right of the agent).
* **Y-axis:** Up-and-down (positive Y is up).
* **Z-axis:** Forward-and-backward (positive Z is further away from the camera).

The answer should be in the format: [x, y, z].
""".strip()

# ==============================================================================
# POST PROMPTS
# ==============================================================================

DISTANCE_POST_PROMPT = """
Please assume the distance is measured from the approximated center of each object, and briefly explain the method used to determine the center.
""".strip()

NA_POST_PROMPT = """
The answer should be a single NUMBER given to one decimal place.
""".strip()

MCA_POST_PROMPT = """
The answer should be the single multiple-choice LETTER answer, formatted as a string.
""".strip()

POST_PROMPT = """
[Hints]
Note that the field-of-view of the camera is {fov} degrees and the image resolution (height and width) is {image_res} pixels.

[Output]
You have to end your response with the answer formatted in a dictionary: {{'answer': <answer>}}. For example, {{'answer': 'Z'}} or {{'answer': 0}} or {{'answer': [0, 0, 0]}}, depending on the question.
If you are not sure about the answer or feel there is insufficient information to answer the question, you can respond with {{'answer': None}}.
""".strip()

# ==============================================================================
# MULTIVIEW PROMPTS (for multi-view questions)
# ==============================================================================

MULTIVIEW_CONTEXT_PROMPT = """
You are given {num_views} views of the same scene from different camera angles.
Use information from all views to answer the following question more accurately.
""".strip()

SINGLE_VIEW_CONTEXT_PROMPT = """
You are given a single view of the scene.
""".strip()

# ==============================================================================
# QUESTION TYPE CATEGORIES
# ==============================================================================

SINGLE_OBJECT_QUESTIONS = [
    'object_size',
    'object_distance_to_camera',
]

PAIR_OBJECT_QUESTIONS = [
    'object_size_comparison_relative',
    'object_size_comparison_absolute',
    'object_pair_distance_center',
    'object_pair_distance_center_w_size',
]

MULTI_OBJECT_QUESTIONS = [
    'object_comparison_absolute_distance',
    'object_comparison_relative_distance',
]

OTHER_OBJECT_QUESTIONS = [
    'object_pair_distance_vector',
]

ALL_QUESTION_TYPES = (
    SINGLE_OBJECT_QUESTIONS + 
    PAIR_OBJECT_QUESTIONS + 
    MULTI_OBJECT_QUESTIONS + 
    OTHER_OBJECT_QUESTIONS
)

# ==============================================================================
# QUESTION TYPE TO TEMPLATE MAPPING
# ==============================================================================

QUESTION_TYPE_TO_TEMPLATE = {
    'object_size': OBJECT_SIZE_TEMPLATE,
    'object_size_comparison_relative': OBJECT_SIZE_COMPARISON_RELATIVE_TEMPLATE,
    'object_size_comparison_absolute': OBJECT_SIZE_COMPARISON_ABSOLUTE_TEMPLATE,
    'object_pair_distance_center': OBJECT_PAIR_DISTANCE_CENTER_TEMPLATE,
    'object_comparison_absolute_distance': OBJECT_COMPARISON_ABSOLUTE_DISTANCE_TEMPLATE,
    'object_comparison_relative_distance': OBJECT_COMPARISON_RELATIVE_DISTANCE_TEMPLATE,
    'object_pair_distance_center_w_size': OBJECT_PAIR_DISTANCE_CENTER_W_SIZE_TEMPLATE,
    'object_distance_to_camera': OBJECT_DISTANCE_TO_CAMERA_TEMPLATE,
    'object_pair_distance_vector': OBJECT_PAIR_DISTANCE_VECTOR_TEMPLATE,
}
