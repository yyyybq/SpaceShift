"""Normalized evaluation question builders shared by both engines.

Example entry:
  {
    "question": "What is the estimated length of the sofa in this video in meters? ...",
    "ground_truth": "1.8",
    "question_type": "object_dimensions",
    "dimension": "length"
  }
"""

DIMENSION_DEFINITION = (
    "Length and width are the two dimensions that define the 'base' of the object. "
    "Of these two base dimensions, let length be the longer and width be the shorter."
)

OBJECT_DIMENSION_TEMPLATE = (
    "What is the estimated {dimension} of the {object_label} in this video in meters? "
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


def build_object_dimension_entry(object_label: str, dimension: str, value: float) -> dict:
    return {
        "question": " ".join([
            OBJECT_DIMENSION_TEMPLATE.format(dimension=dimension, object_label=object_label),
            NA_POST_PROMPT,
            POST_PROMPT,
        ]),
        "ground_truth": str(round(value, 1)),
        "question_type": "object_dimensions",
        "question_family": "size",
        "dimension": dimension,
    }


def build_object_distance_entry(object_label: str, distance: float) -> dict:
    return {
        "question": " ".join([
            OBJECT_DISTANCE_TEMPLATE.format(object_label=object_label),
            DISTANCE_POST_PROMPT,
            NA_POST_PROMPT,
            POST_PROMPT,
        ]),
        "ground_truth": str(round(distance, 1)),
        "question_type": "object_distance_to_camera",
        "question_family": "camera_distance",
        "dimension": None,
    }


def build_object_pair_distance_entry(object_label_a: str, object_label_b: str, distance: float) -> dict:
    return {
        "question": " ".join([
            OBJECT_PAIR_DISTANCE_TEMPLATE.format(
                object_label_a=object_label_a,
                object_label_b=object_label_b,
            ),
            DISTANCE_POST_PROMPT,
            NA_POST_PROMPT,
            POST_PROMPT,
        ]),
        "ground_truth": str(round(distance, 1)),
        "question_type": "object_pair_distance_center",
        "question_family": "pair_distance",
        "dimension": None,
    }
