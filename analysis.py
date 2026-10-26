from io import BytesIO
from pathlib import Path
import re
import cv2
import numpy as np
import pymupdf
import pytesseract
import base64

from PIL import (
    Image,
    ImageChops,
    ImageEnhance,
    ImageFilter,
    ImageOps,
    ImageStat
)
import tensorflow as tf

# --------------------------------------------------
# Tesseract configuration
# --------------------------------------------------

WINDOWS_TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if WINDOWS_TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(
        WINDOWS_TESSERACT_PATH
    )

#ML
BASE_FOLDER = Path(
    __file__
).resolve().parent

ML_MODEL_PATH = (
    BASE_FOLDER
    / "models"
    / "forgery_model.keras"
)

_loaded_ml_model = None
# --------------------------------------------------
# Rules for the ten supported document categories
# --------------------------------------------------

DOCUMENT_RULES = {
    "pan_card": {
        "strong": [
            "permanent account number",
            "income tax department"
        ],
        "supporting": [
            "government of india",
            "govt of india",
            "father",
            "date of birth",
            "signature",
            "income tax"
        ]
    },

    "aadhaar_card": {
        "strong": [
            "aadhaar",
            "aadhar",
            "unique identification authority",
            "mera aadhaar meri pehchan"
        ],
        "supporting": [
            "government of india",
            "govt of india",
            "date of birth",
            "year of birth",
            "male",
            "female",
            "address",
            "vid"
        ]
    },

    "marksheet": {
        "strong": [
            "marksheet",
            "mark sheet",
            "statement of marks",
            "grade card",
            "academic transcript"
        ],
        "supporting": [
            "marks obtained",
            "maximum marks",
            "max marks",
            "total marks",
            "roll number",
            "roll no",
            "enrollment number",
            "subject code",
            "grade",
            "semester",
            "result"
        ]
    },

    "educational_certificate": {
        "strong": [
            "degree certificate",
            "provisional certificate",
            "educational certificate",
            "certificate of completion"
        ],
        "supporting": [
            "university",
            "college",
            "awarded",
            "degree",
            "successfully completed",
            "registrar",
            "course"
        ]
    },

    "college_id": {
        "strong": [
            "student identity card",
            "student id card",
            "college identity card",
            "identity card"
        ],
        "supporting": [
            "college",
            "university",
            "student",
            "roll number",
            "roll no",
            "enrollment number",
            "course",
            "valid upto",
            "valid up to"
        ]
    },

    "birth_certificate": {
        "strong": [
            "birth certificate",
            "certificate of birth",
            "registration of birth"
        ],
        "supporting": [
            "date of birth",
            "place of birth",
            "name of mother",
            "name of father",
            "mother",
            "father",
            "registrar",
            "registration number",
            "registration no"
        ]
    },

    "income_certificate": {
        "strong": [
            "income certificate",
            "certificate of income"
        ],
        "supporting": [
            "annual income",
            "family income",
            "financial year",
            "revenue officer",
            "tehsildar",
            "applicant",
            "total income"
        ]
    },

    "experience_certificate": {
        "strong": [
            "experience certificate",
            "certificate of experience",
            "work experience certificate"
        ],
        "supporting": [
            "worked with",
            "employed",
            "designation",
            "joining date",
            "date of joining",
            "employment",
            "organization",
            "company"
        ]
    },

    "medical_certificate": {
        "strong": [
            "medical certificate",
            "certificate of fitness",
            "medical fitness certificate"
        ],
        "supporting": [
            "patient",
            "doctor",
            "hospital",
            "medical practitioner",
            "diagnosis",
            "registration number",
            "registration no"
        ]
    },

    "domicile_certificate": {
        "strong": [
            "domicile certificate",
            "certificate of domicile",
            "residence certificate"
        ],
        "supporting": [
            "permanent resident",
            "resident of",
            "district",
            "state",
            "tehsildar",
            "revenue officer",
            "applicant"
        ]
    }
}


# --------------------------------------------------
# Load image or PDF
# --------------------------------------------------

def load_document_image(file_path):
    """
    Load an image or convert the first page of a PDF
    into a high-resolution image.
    """

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        pdf_document = pymupdf.open(file_path)

        try:
            if pdf_document.page_count == 0:
                raise ValueError(
                    "The selected PDF has no pages."
                )

            first_page = pdf_document.load_page(0)

            zoom_matrix = pymupdf.Matrix(2, 2)

            page_pixmap = first_page.get_pixmap(
                matrix=zoom_matrix,
                alpha=False
            )

            image_bytes = page_pixmap.tobytes(
                "png"
            )

            document_image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            document_format = "PDF"

        finally:
            pdf_document.close()

    else:
        opened_image = Image.open(file_path)

        document_format = (
            opened_image.format
            or extension.replace(".", "").upper()
        )

        document_image = opened_image.convert(
            "RGB"
        )

    return document_image, document_format


# --------------------------------------------------
# OCR preprocessing
# --------------------------------------------------

def prepare_image_for_ocr(image):
    """
    Enlarge and enhance a document image for OCR.
    """

    grayscale_image = ImageOps.grayscale(
        image
    )

    # Enlarge small images while preserving aspect ratio.
    if grayscale_image.width < 1200:
        scale = (
            1200
            / grayscale_image.width
        )

        new_width = round(
            grayscale_image.width
            * scale
        )

        new_height = round(
            grayscale_image.height
            * scale
        )

        grayscale_image = (
            grayscale_image.resize(
                (
                    new_width,
                    new_height
                ),
                Image.Resampling.LANCZOS
            )
        )

    grayscale_image = ImageOps.autocontrast(
        grayscale_image
    )

    grayscale_image = ImageEnhance.Contrast(
        grayscale_image
    ).enhance(
        2
    )

    grayscale_image = grayscale_image.filter(
        ImageFilter.SHARPEN
    )

    return grayscale_image


def extract_text(image):
    """
    Test multiple orientations and Tesseract page
    segmentation modes, then return the OCR result
    with the highest confidence and readability.
    """

    orientation_candidates = [
        (
            "original",
            image
        ),

        (
            "rotate_90_counterclockwise",
            image.rotate(
                90,
                expand=True
            )
        ),

        (
            "rotate_90_clockwise",
            image.rotate(
                -90,
                expand=True
            )
        ),

        (
            "rotate_180",
            image.rotate(
                180,
                expand=True
            )
        )
    ]

    page_segmentation_modes = (
        6,
        11
    )

    # Use identifying phrases from all ten supported
    # document categories when selecting orientation.
    document_phrases = tuple(
        dict.fromkeys(
            phrase
            for rule in DOCUMENT_RULES.values()
            for phrase_group in (
                "strong",
                "supporting"
            )
            for phrase in rule[
                phrase_group
            ]
        )
    )

    best_text = ""
    best_score = -1

    for (
        orientation_name,
        oriented_image
    ) in orientation_candidates:
        prepared_image = prepare_image_for_ocr(
            oriented_image
        )

        for psm in page_segmentation_modes:
            try:
                ocr_data = (
                    pytesseract.image_to_data(
                        prepared_image,
                        lang="eng",
                        config=(
                            f"--oem 3 --psm {psm}"
                        ),
                        output_type=(
                            pytesseract.Output.DICT
                        )
                    )
                )

            except pytesseract.TesseractError:
                continue

            readable_words = []
            confidence_total = 0.0
            line_groups = []
            current_line_key = None
            current_line_words = []

            item_count = len(
                ocr_data["text"]
            )

            for item_index in range(
                item_count
            ):
                word = (
                    ocr_data["text"][
                        item_index
                    ]
                    .strip()
                )

                if not word:
                    continue

                line_key = (
                    ocr_data["block_num"][
                        item_index
                    ],
                    ocr_data["par_num"][
                        item_index
                    ],
                    ocr_data["line_num"][
                        item_index
                    ]
                )

                if (
                    current_line_key is not None
                    and line_key
                    != current_line_key
                ):
                    if current_line_words:
                        line_groups.append(
                            " ".join(
                                current_line_words
                            )
                        )

                    current_line_words = []

                current_line_key = line_key

                current_line_words.append(
                    word
                )

                if re.search(
                    r"[A-Za-z0-9]",
                    word
                ):
                    readable_words.append(
                        word
                    )

                    try:
                        confidence = float(
                            ocr_data["conf"][
                                item_index
                            ]
                        )

                    except (
                        TypeError,
                        ValueError
                    ):
                        confidence = 0.0

                    if confidence > 0:
                        confidence_total += (
                            confidence
                        )

            if current_line_words:
                line_groups.append(
                    " ".join(
                        current_line_words
                    )
                )

            candidate_text = "\n".join(
                line_groups
            ).strip()

            normalized_candidate = (
                normalize_ocr_text(
                    candidate_text
                )
            )

            phrase_matches = sum(
                1
                for phrase
                in document_phrases
                if normalize_ocr_text(
                    phrase
                )
                in normalized_candidate
            )

            readable_character_count = sum(
                len(
                    re.sub(
                        r"[^A-Za-z0-9]",
                        "",
                        word
                    )
                )
                for word in readable_words
            )

            meaningful_word_count = sum(
                1
                for word in readable_words
                if len(
                    re.sub(
                        r"[^A-Za-z0-9]",
                        "",
                        word
                    )
                ) >= 3
            )

            if readable_words:
                average_confidence = (
                    confidence_total
                    / len(
                        readable_words
                    )
                )

            else:
                average_confidence = 0

            uppercase_candidate = (
                candidate_text.upper()
            )

            pan_pattern_found = re.search(
                r"\b[A-Z]{5}"
                r"[\s-]*[0-9]{4}"
                r"[\s-]*[A-Z]\b",
                uppercase_candidate
            )

            aadhaar_pattern_found = re.search(
                r"\b[2-9][0-9]{3}"
                r"[\s-]*[0-9]{4}"
                r"[\s-]*[0-9]{4}\b",
                candidate_text
            )

            # Category phrases and valid identification
            # patterns receive the highest priority.
            # This prevents long sideways garbage text
            # from winning orientation selection.
            candidate_score = (
                phrase_matches * 10000
                + meaningful_word_count * 15
                + average_confidence * 20
                + readable_character_count * 0.20
            )

            if pan_pattern_found:
                candidate_score += 15000

            if aadhaar_pattern_found:
                candidate_score += 15000

            if candidate_score > best_score:
                best_score = candidate_score
                best_text = candidate_text

    return best_text.strip()


# --------------------------------------------------
# OCR text normalization
# --------------------------------------------------

def normalize_ocr_text(text):
    """
    Normalize OCR text so matching is unaffected by
    uppercase letters, punctuation or extra spaces.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# --------------------------------------------------
# Document-type verification
# --------------------------------------------------

def verify_document_type(
    extracted_text,
    selected_type
):
    """
    Verify that the uploaded document matches the
    category selected by the user.
    """

    rule = DOCUMENT_RULES.get(
        selected_type
    )

    if rule is None:
        return {
            "type_verified": False,
            "match_percentage": 0,
            "matched_keywords": [],
            "strong_matches": [],
            "supporting_matches": [],
            "verification_message": (
                "The selected document category "
                "is not supported."
            )
        }

    normalized_text = normalize_ocr_text(
        extracted_text
    )

    matched_strong = []
    matched_supporting = []

    # Check strong document-identifying phrases
    for phrase in rule["strong"]:
        normalized_phrase = normalize_ocr_text(
            phrase
        )

        if normalized_phrase in normalized_text:
            matched_strong.append(
                phrase
            )

    # Check supporting document fields
    for phrase in rule["supporting"]:
        normalized_phrase = normalize_ocr_text(
            phrase
        )

        if normalized_phrase in normalized_text:
            matched_supporting.append(
                phrase
            )

    # PAN-number pattern:
    # five letters + four digits + one letter
    if selected_type == "pan_card":
        uppercase_text = extracted_text.upper()

        pan_number_match = re.search(
            r"\b[A-Z]{5}\s*[0-9]{4}\s*[A-Z]\b",
            uppercase_text
        )

        if (
            pan_number_match
            and "PAN number format"
            not in matched_strong
        ):
            matched_strong.append(
                "PAN number format"
            )

    # Aadhaar-number pattern:
    # twelve digits, usually shown in groups of four
    if selected_type == "aadhaar_card":
        aadhaar_number_match = re.search(
            r"\b[2-9][0-9]{3}\s*[0-9]{4}\s*[0-9]{4}\b",
            extracted_text
        )

        if (
            aadhaar_number_match
            and "Aadhaar number format"
            not in matched_strong
        ):
            matched_strong.append(
                "Aadhaar number format"
            )

    matched_keywords = (
        matched_strong
        + matched_supporting
    )

    total_indicators = (
        len(rule["strong"])
        + len(rule["supporting"])
    )

    if total_indicators == 0:
        match_percentage = 0
    else:
        match_percentage = round(
            (
                len(matched_keywords)
                / total_indicators
            ) * 100
        )

    # Require both:
    # Standard verification:
    # Require one strong category identifier and
    # at least one supporting document field.
    standard_verification = (
        len(matched_strong) >= 1
        and len(matched_supporting) >= 1
    )

    # Aadhaar OCR fallback:
    # A manipulated or low-quality Aadhaar image may
    # lose its title while retaining its identifying
    # demographic fields.
    aadhaar_birth_field_found = any(
        phrase in matched_supporting
        for phrase in (
            "date of birth",
            "year of birth"
        )
    )

    aadhaar_gender_field_found = any(
        phrase in matched_supporting
        for phrase in (
            "male",
            "female"
        )
    )

    aadhaar_address_field_found = (
        "address"
        in matched_supporting
    )

    aadhaar_supporting_fallback = (
        selected_type == "aadhaar_card"
        and aadhaar_birth_field_found
        and aadhaar_gender_field_found
        and aadhaar_address_field_found
    )

    college_name_found = any(
        phrase in matched_supporting
        for phrase in (
            "college",
            "university"
        )
    )

    college_student_or_course_found = any(
        phrase in matched_supporting
        for phrase in (
            "student",
            "course"
        )
    )

    college_number_field_found = any(
        phrase in matched_supporting
        for phrase in (
            "roll number",
            "roll no",
            "enrollment number"
        )
    )

    marksheet_indicators_found = any(
        phrase in normalized_text
        for phrase in (
            "marksheet",
            "mark sheet",
            "statement of marks",
            "marks obtained",
            "maximum marks",
            "total marks",
            "subject code",
            "grade card"
        )
    )

    college_id_supporting_fallback = (
        selected_type == "college_id"
        and college_name_found
        and college_student_or_course_found
        and college_number_field_found
        and len(matched_supporting) >= 4
        and not marksheet_indicators_found
    )

    type_verified = (
        standard_verification
        or aadhaar_supporting_fallback
        or college_id_supporting_fallback
    )

    if type_verified:
        if aadhaar_supporting_fallback:
            verification_message = (
                "The Aadhaar title was not clearly read, "
                "but the date-of-birth, gender and address "
                "fields support the selected document type."
            )

        elif college_id_supporting_fallback:
            verification_message = (
                "The College ID title was not clearly "
                "read, but the institution, course and "
                "roll-number fields support the selected "
                "document type."
            )

        else:
            verification_message = (
                "The document contains a category-specific "
                "identifier and supporting fields matching "
                "the selected document type."
            )

    elif not extracted_text.strip():
        verification_message = (
            "No readable text was found. Upload a clearer "
            "and properly aligned document."
        )

    elif len(matched_strong) == 0:
        verification_message = (
            "The required document title, identification "
            "number or identifying phrase was not detected."
        )

    else:
        verification_message = (
            "A category identifier was detected, but "
            "there were not enough supporting fields "
            "to verify the document type."
        )

    return {
        "type_verified": type_verified,
        "match_percentage": match_percentage,
        "matched_keywords": matched_keywords,
        "strong_matches": matched_strong,
        "supporting_matches": matched_supporting,
        "verification_message": verification_message
    }


# --------------------------------------------------
# Initial forgery score
# --------------------------------------------------

def calculate_forgery_score(original_image):
    """
    Calculate an initial JPEG recompression-difference
    risk score.
    """

    temporary_image = BytesIO()

    original_image.save(
        temporary_image,
        format="JPEG",
        quality=90
    )

    temporary_image.seek(0)

    compressed_image = Image.open(
        temporary_image
    ).convert("RGB")

    difference_image = ImageChops.difference(
        original_image,
        compressed_image
    )

    statistics = ImageStat.Stat(
        difference_image
    )

    average_difference = (
        sum(statistics.mean) / 3
    )

    risk_score = min(
        round(average_difference * 10),
        100
    )

    return risk_score

def divide_into_blocks(array, rows=4, columns=4):
    """
    Divide an image array into smaller regions.
    """

    height, width = array.shape[:2]

    block_height = max(
        height // rows,
        1
    )

    block_width = max(
        width // columns,
        1
    )

    blocks = []

    for row in range(rows):
        for column in range(columns):
            start_y = row * block_height
            start_x = column * block_width

            end_y = (
                height
                if row == rows - 1
                else (row + 1) * block_height
            )

            end_x = (
                width
                if column == columns - 1
                else (column + 1) * block_width
            )

            block = array[
                start_y:end_y,
                start_x:end_x
            ]

            if block.size > 0:
                blocks.append(block)

    return blocks


def calculate_edge_inconsistency(image):
    """
    Measure differences in edge density between
    document regions.
    """

    image_array = np.array(
        image.convert("RGB")
    )

    grayscale = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        grayscale,
        80,
        180
    )

    blocks = divide_into_blocks(
        edges
    )

    edge_densities = []

    for block in blocks:
        density = np.mean(
            block > 0
        )

        edge_densities.append(
            density
        )

    if not edge_densities:
        return 0

    inconsistency = np.std(
        edge_densities
    )

    score = round(
        inconsistency * 500
    )

    return int(
        np.clip(score, 0, 100)
    )


def calculate_noise_inconsistency(image):
    """
    Measure variation in local image noise.
    """

    image_array = np.array(
        image.convert("RGB")
    )

    grayscale = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )

    noise = np.abs(
        grayscale - blurred
    )

    blocks = divide_into_blocks(
        noise
    )

    noise_levels = []

    for block in blocks:
        noise_levels.append(
            float(np.mean(block))
        )

    if not noise_levels:
        return 0

    inconsistency = np.std(
        noise_levels
    )

    score = round(
        inconsistency * 12
    )

    return int(
        np.clip(score, 0, 100)
    )


def calculate_blur_inconsistency(image):
    """
    Measure differences in sharpness between
    document regions.
    """

    image_array = np.array(
        image.convert("RGB")
    )

    grayscale = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    blocks = divide_into_blocks(
        grayscale
    )

    sharpness_values = []

    for block in blocks:
        laplacian = cv2.Laplacian(
            block,
            cv2.CV_64F
        )

        sharpness_values.append(
            float(laplacian.var())
        )

    if not sharpness_values:
        return 0

    average_sharpness = np.mean(
        sharpness_values
    )

    sharpness_variation = np.std(
        sharpness_values
    )

    coefficient = (
        sharpness_variation
        / (average_sharpness + 1)
    )

    score = round(
        coefficient * 35
    )

    return int(
        np.clip(score, 0, 100)
    )
def normalize_forensic_image(
        
        image,
        target_width=1200
    ):
        """
        Resize a document to a consistent width while
        preserving its aspect ratio.
        """

        image = image.convert("RGB")

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                "Invalid document image dimensions."
            )

        scale = target_width / width

        target_height = max(
            round(height * scale),
            1
        )

        normalized_image = image.resize(
            (
                target_width,
                target_height
            ),
            Image.Resampling.LANCZOS
        )

        return normalized_image
def generate_suspicious_heatmap(image):
        """
        Generate a visual attention map from local image
        residuals and return it as a browser-ready data URL.
        """

        normalized_image = normalize_forensic_image(
            image
        )

        rgb_array = np.array(
            normalized_image.convert("RGB")
        )

        grayscale = cv2.cvtColor(
            rgb_array,
            cv2.COLOR_RGB2GRAY
        )

        blurred = cv2.GaussianBlur(
            grayscale,
            (9, 9),
            0
        )

        residual = cv2.absdiff(
            grayscale,
            blurred
        )

        residual = cv2.normalize(
            residual,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Reduce minor background noise
        _, thresholded = cv2.threshold(
            residual,
            35,
            255,
            cv2.THRESH_TOZERO
        )

        heatmap = cv2.applyColorMap(
            thresholded,
            cv2.COLORMAP_JET
        )

        original_bgr = cv2.cvtColor(
            rgb_array,
            cv2.COLOR_RGB2BGR
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.65,
            heatmap,
            0.35,
            0
        )

        success, encoded_image = cv2.imencode(
            ".png",
            overlay
        )

        if not success:
            return None

        encoded_text = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")

        return (
            "data:image/png;base64,"
            + encoded_text
        )

def analyze_forensic_signals(image):

    """
    Combine multiple forensic indicators into
    one risk score.
    """
    image = normalize_forensic_image(
        image
    )

    compression_score = calculate_forgery_score(
        image
    )

    edge_score = calculate_edge_inconsistency(
        image
    )

    noise_score = calculate_noise_inconsistency(
        image
    )

    blur_score = calculate_blur_inconsistency(
        image
    )

    combined_score = round(
        compression_score * 0.40
        + edge_score * 0.20
        + noise_score * 0.25
        + blur_score * 0.15
    )

    combined_score = int(
        np.clip(
            combined_score,
            0,
            100
        )
    )

    signals = [
        {
            "name": "Compression inconsistency",
            "score": compression_score,
            "description": (
                "Measures changes produced by JPEG "
                "recompression."
            )
        },
        {
            "name": "Edge inconsistency",
            "score": edge_score,
            "description": (
                "Measures unusual edge-density differences "
                "between document regions."
            )
        },
        {
            "name": "Noise inconsistency",
            "score": noise_score,
            "description": (
                "Measures variations in local image noise."
            )
        },
        {
            "name": "Blur inconsistency",
            "score": blur_score,
            "description": (
                "Measures differences in sharpness between "
                "document regions."
            )
        }
    ]

    return {
        "risk_score": combined_score,
        "signals": signals
    }

def assess_image_quality(image):
    """
    Determine whether the document image has enough
    resolution and sharpness for forensic analysis.
    """

    width, height = image.size

    image_array = np.array(
        image.convert("RGB")
    )

    grayscale = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    sharpness = cv2.Laplacian(
        grayscale,
        cv2.CV_64F
    ).var()

    issues = []
    deductions = 0

    # General minimum suitable for cards and certificates
    total_pixels = (
        width
        * height
    )

    shortest_side = min(
        width,
        height
    )

    # Accept portrait, landscape and square documents.
    # Require enough total image information for analysis.
    resolution_suitable = (
        shortest_side >= 300
        and total_pixels >= 250000
    )

    if not resolution_suitable:
        issues.append(
            "Document resolution is too low."
        )

        deductions += 60

    if sharpness < 25:
        issues.append(
            "Document image is severely blurred."
        )

        deductions += 40

    elif sharpness < 60:
        issues.append(
            "Document image has limited sharpness."
        )

        deductions += 20

    quality_score = max(
        100 - deductions,
        0
    )

    suitable_for_analysis = (
        resolution_suitable
        and sharpness >= 25
    )

    return {
        "suitable": suitable_for_analysis,
        "quality_score": quality_score,
        "width": width,
        "height": height,
        "sharpness": round(sharpness, 2),
        "issues": issues
}
# --------------------------------------------------
# Complete document-analysis pipeline
# --------------------------------------------------
def get_ml_model():
    """
    Load the trained model only once.
    """

    global _loaded_ml_model

    if _loaded_ml_model is not None:
        return _loaded_ml_model

    if not ML_MODEL_PATH.exists():
        return None

    _loaded_ml_model = (
        tf.keras.models.load_model(
            ML_MODEL_PATH
        )
    )

    return _loaded_ml_model


def predict_ml_forgery(image):
    """
    Return the model's predicted forged probability
    as a percentage.

    Preprocessing exactly matches train_model.py.
    """

    model = get_ml_model()

    if model is None:
        return None

    rgb_image = image.convert(
        "RGB"
    )

    image_tensor = tf.convert_to_tensor(
        np.array(rgb_image),
        dtype=tf.uint8
    )

    image_tensor = tf.image.resize(
        image_tensor,
        (256, 256)
    )

    image_tensor = tf.cast(
        image_tensor,
        tf.float32
    )

    image_batch = tf.expand_dims(
        image_tensor,
        axis=0
    )

    prediction = model.predict(
        image_batch,
        verbose=0
    )

    forged_probability = float(
        prediction[0][0]
    )

    forged_score = round(
        forged_probability * 100
    )

    return int(
        np.clip(
            forged_score,
            0,
            100
        )
    )


def analyze_document(
    file_path,
    selected_type
):
    """
    First verify the document category. Only verified
    document categories receive a forgery score.
    """

    original_image, document_format = (
        load_document_image(file_path)
    )

    quality_result = assess_image_quality(
        original_image
    )

    extracted_text = extract_text(
        original_image
    )

    type_result = verify_document_type(
        extracted_text,
        selected_type
    )

    result = {
        "width": original_image.width,
        "height": original_image.height,
        "format": document_format,
        "extracted_text": extracted_text,
        "type_verified": type_result[
            "type_verified"
        ],
        "match_percentage": type_result[
            "match_percentage"
        ],
        "matched_keywords": type_result[
            "matched_keywords"
        ],
        "strong_matches": type_result[
            "strong_matches"
        ],
        "supporting_matches": type_result[
            "supporting_matches"
        ],
        "verification_message": type_result[
            "verification_message"
        ],
        "quality_suitable": quality_result[
            "suitable"
        ],
        "quality_score": quality_result[
            "quality_score"
        ],
        "quality_issues": quality_result[
            "issues"
        ],
        "sharpness": quality_result[
            "sharpness"
        ]
    }

    # Block forgery analysis when the selected
    # document category cannot be verified
    if not type_result["type_verified"]:
        result.update({
            "analysis_performed": False,
            "risk_score": None,
            "authenticity_score": None,
            "risk_level": "Not analyzed",
            "forensic_signals": [],
            "suspicious_heatmap": None,
            "heuristic_risk_score": None,
            "ml_forgery_score": None,
            "ml_model_used": False,
            "result_message": (
                "Forgery analysis was not performed "
                "because the uploaded file could not "
                "be verified as the selected document "
                "category."
            )
        })

        return result
    if not quality_result["suitable"]:
        result.update({
            "analysis_performed": False,
            "risk_score": None,
            "authenticity_score": None,
            "risk_level": "Insufficient quality",
            "forensic_signals": [],
            "suspicious_heatmap": None,
            "heuristic_risk_score": None,
            "ml_forgery_score": None,
            "ml_model_used": False,
            "result_message": (
                "The document category was verified, "
                "but the image quality is insufficient "
                "for reliable forgery analysis."
            )
        })

        return result
    def normalize_forensic_image(
            image,
            target_width=1200
        ):
            """
            Resize accepted documents to a consistent width
            while preserving their aspect ratio.
            """

            image = image.convert("RGB")

            width, height = image.size

            scale = target_width / width

            target_height = max(
                round(height * scale),
                1
            )

            normalized_image = image.resize(
                (
                    target_width,
                    target_height
                ),
                Image.Resampling.LANCZOS
            )

            return normalized_image
    forensic_result = analyze_forensic_signals(
        original_image
    )

    suspicious_heatmap = generate_suspicious_heatmap(
        original_image
    )

    heuristic_risk_score = forensic_result[
        "risk_score"
    ]

    ml_forgery_score = predict_ml_forgery(
        original_image
    )

    forensic_signals = list(
        forensic_result["signals"]
    )

    ml_forgery_threshold = 19
    ml_flagged_as_suspicious = False
    evidence_disagreement = False
    if ml_forgery_score is not None:
        ml_flagged_as_suspicious = (
            ml_forgery_score
            >= ml_forgery_threshold
        )

        forensic_signals.append({
            "name": "ML forged probability",
            "score": ml_forgery_score,
            "description": (
                "Probability produced by the trained "
                "mixed real-and-synthetic document "
                "classification model."
            )
        })

        evidence_disagreement = (
            ml_forgery_score >= 55
            and heuristic_risk_score < 25
        )

        if evidence_disagreement:
            # Reduce ML influence when the CNN strongly
            # disagrees with the forensic indicators.
            combined_risk_score = round(
                heuristic_risk_score * 0.65
                + ml_forgery_score * 0.35
            )

        else:
            combined_risk_score = round(
                heuristic_risk_score * 0.45
                + ml_forgery_score * 0.55
            )

        if ml_flagged_as_suspicious:
            risk_score = max(
                combined_risk_score,
                25
            )

        else:
            risk_score = combined_risk_score

    else:
        risk_score = heuristic_risk_score

    risk_score = int(
        np.clip(
            risk_score,
            0,
            100
        )
    )

    authenticity_score = (
        100
        - risk_score
    )

    if risk_score < 25:
        risk_level = "Low"

        result_message = (
            "No strong document-manipulation "
            "indicators were detected."
        )

    elif risk_score < 55:
        risk_level = "Medium"

        if evidence_disagreement:
            result_message = (
                "The AI model detected a strong potential "
                "forgery pattern, but the image-forensic "
                "indicators were comparatively weak. "
                "Manual verification is required."
            )

        elif ml_flagged_as_suspicious:
            result_message = (
                "The AI model detected a potential "
                "forgery pattern. Manual verification "
                "is recommended."
            )

        else:
            result_message = (
                "Some forensic inconsistencies were "
                "detected. Manual verification is "
                "recommended."
            )

    else:
        risk_level = "High"

        result_message = (
            "Strong document-manipulation indicators "
            "were detected. Treat the document as "
            "suspicious until manually verified."
        )

    result.update({
        "analysis_performed": True,
        "risk_score": risk_score,
        "authenticity_score": authenticity_score,
        "risk_level": risk_level,
        "forensic_signals": forensic_signals,
        "heuristic_risk_score": heuristic_risk_score,
        "ml_forgery_score": ml_forgery_score,
        "ml_model_used": (
            ml_forgery_score is not None
        ),
        "suspicious_heatmap": suspicious_heatmap,
        "result_message": result_message
    })

    return result