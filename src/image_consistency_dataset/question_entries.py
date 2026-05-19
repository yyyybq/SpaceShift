"""Image-variant question templates for the image consistency benchmark.

Same questions as the video benchmark but with "in this video" → "in this image".
"""

DIMENSION_DEFINITION = (
    "Length and width are the two dimensions that define the 'base' of the object. "
    "Of these two base dimensions, let length be the longer and width be the shorter."
)

OBJECT_DIMENSION_TEMPLATE = (
    "What is the estimated {dimension} of the {object_label} in this image in meters? "
    + DIMENSION_DEFINITION
)

OBJECT_DISTANCE_TEMPLATE = (
    "What is the distance from the camera to the {object_label} in meters?"
)

OBJECT_PAIR_DISTANCE_TEMPLATE = (
    "What is the absolute distance between {object_label_a} and {object_label_b} in meters?"
)

DISTANCE_POST_PROMPT = (
    "Please assume the distance is measured from the approximated center of each object, "
    "and briefly explain the method used to determine the center."
)

NA_POST_PROMPT = "The answer should be a single NUMBER given to one decimal place."

POST_PROMPT = (
    "[Output]\n"
    "You have to end your response with the answer formatted in a dictionary: {{'answer': <answer>}}. "
    "For example, {{'answer': 0}}."
)


def rewrite_question_for_image(question_text: str) -> str:
    """Replace 'in this video' with 'in this image' in an existing question."""
    return question_text.replace("in this video", "in this image")


def build_image_dimension_question(object_label: str, dimension: str) -> str:
    return " ".join([
        OBJECT_DIMENSION_TEMPLATE.format(dimension=dimension, object_label=object_label),
        NA_POST_PROMPT,
        POST_PROMPT,
    ])


def build_image_distance_question(object_label: str) -> str:
    return " ".join([
        OBJECT_DISTANCE_TEMPLATE.format(object_label=object_label),
        DISTANCE_POST_PROMPT,
        NA_POST_PROMPT,
        POST_PROMPT,
    ])


def build_image_pair_distance_question(object_label_a: str, object_label_b: str) -> str:
    return " ".join([
        OBJECT_PAIR_DISTANCE_TEMPLATE.format(
            object_label_a=object_label_a,
            object_label_b=object_label_b,
        ),
        DISTANCE_POST_PROMPT,
        NA_POST_PROMPT,
        POST_PROMPT,
    ])
