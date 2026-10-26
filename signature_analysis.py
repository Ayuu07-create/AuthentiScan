import cv2
import numpy as np


def prepare_signature(file_path):
    """
    Convert a signature image into a normalized
    black-and-white representation.
    """

    image = cv2.imread(
        str(file_path)
    )

    if image is None:
        raise ValueError(
            "The signature image could not be opened."
        )

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    grayscale = cv2.GaussianBlur(
        grayscale,
        (3, 3),
        0
    )

    binary = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10
    )

    points = cv2.findNonZero(
        binary
    )

    if points is None:
        raise ValueError(
            "No readable signature was detected."
        )

    x, y, width, height = cv2.boundingRect(
        points
    )

    cropped = binary[
        y:y + height,
        x:x + width
    ]

    normalized = cv2.resize(
        cropped,
        (400, 160),
        interpolation=cv2.INTER_AREA
    )

    return normalized


def calculate_pixel_similarity(
    first_signature,
    second_signature
):
    """
    Compare normalized signature shapes using
    correlation.
    """

    first_float = first_signature.astype(
        np.float32
    )

    second_float = second_signature.astype(
        np.float32
    )

    correlation = cv2.matchTemplate(
        first_float,
        second_float,
        cv2.TM_CCOEFF_NORMED
    )[0][0]

    correlation = max(
        float(correlation),
        0
    )

    return round(
        correlation * 100
    )


def calculate_feature_similarity(
    first_signature,
    second_signature
):
    """
    Compare local signature features using ORB.
    """

    orb = cv2.ORB_create(
        nfeatures=500
    )

    keypoints_one, descriptors_one = (
        orb.detectAndCompute(
            first_signature,
            None
        )
    )

    keypoints_two, descriptors_two = (
        orb.detectAndCompute(
            second_signature,
            None
        )
    )

    if (
        descriptors_one is None
        or descriptors_two is None
    ):
        return 0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(
        descriptors_one,
        descriptors_two
    )

    if not matches:
        return 0

    good_matches = [
        match
        for match in matches
        if match.distance < 55
    ]

    possible_matches = max(
        min(
            len(keypoints_one),
            len(keypoints_two)
        ),
        1
    )

    similarity = (
        len(good_matches)
        / possible_matches
    ) * 100

    return int(
        np.clip(
            round(similarity),
            0,
            100
        )
    )


def compare_signatures(
    questioned_path,
    reference_path
):
    """
    Compare questioned and genuine-reference
    signature images.
    """

    questioned = prepare_signature(
        questioned_path
    )

    reference = prepare_signature(
        reference_path
    )

    pixel_score = calculate_pixel_similarity(
        questioned,
        reference
    )

    feature_score = calculate_feature_similarity(
        questioned,
        reference
    )

    similarity_score = round(
        pixel_score * 0.55
        + feature_score * 0.45
    )

    if similarity_score >= 70:
        status = "Likely match"

        message = (
            "The questioned signature has strong "
            "visual similarity to the reference."
        )

    elif similarity_score >= 45:
        status = "Manual review"

        message = (
            "The signatures have partial similarity "
            "and require manual review."
        )

    else:
        status = "Likely mismatch"

        message = (
            "The questioned signature has low visual "
            "similarity to the reference."
        )

    return {
        "similarity_score": similarity_score,
        "pixel_score": pixel_score,
        "feature_score": feature_score,
        "status": status,
        "message": message
    }