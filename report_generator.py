from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm

from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)


def generate_analysis_report(analysis):
    """
    Generate a PDF report in memory.
    """

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Document Verification Report"
    )

    styles = getSampleStyleSheet()

    report_elements = []

    report_elements.append(
        Paragraph(
            "Document Verification Report",
            styles["Title"]
        )
    )

    report_elements.append(
        Spacer(1, 10 * mm)
    )

    if analysis["document_verified"]:
        verification_status = (
            "Verified document category"
        )
    else:
        verification_status = (
            "Document category not verified"
        )

    authenticity_score = (
        f'{analysis["authenticity_score"]}%'
        if analysis["authenticity_score"] is not None
        else "Not analyzed"
    )

    report_data = [
        [
            "Field",
            "Result"
        ],
        [
            "Original filename",
            analysis["original_filename"]
        ],
        [
            "Selected document type",
            analysis["document_type"]
        ],
        [
            "Document-type status",
            verification_status
        ],
        [
            "Authenticity score",
            authenticity_score
        ],
        [
            "Risk level",
            analysis["risk_level"]
        ],
        [
            "Analysis date",
            analysis["created_at"]
        ]
    ]

    report_table = Table(
        report_data,
        colWidths=[
            55 * mm,
            110 * mm
        ]
    )

    report_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#0F172A")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "BACKGROUND",
                (0, 1),
                (0, -1),
                colors.HexColor("#E2E8F0")
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#94A3B8")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    report_elements.append(
        report_table
    )

    report_elements.append(
        Spacer(1, 10 * mm)
    )

    report_elements.append(
        Paragraph(
            "Extracted OCR Text",
            styles["Heading2"]
        )
    )

    extracted_text = (
        analysis["extracted_text"]
        or "No readable OCR text was extracted."
    )

    safe_text = escape(
        extracted_text
    ).replace(
        "\n",
        "<br/>"
    )

    report_elements.append(
        Paragraph(
            safe_text,
            styles["BodyText"]
        )
    )

    report_elements.append(
        Spacer(1, 10 * mm)
    )

    report_elements.append(
        Paragraph(
            (
                "<b>Important:</b> This report is an "
                "AI-assisted forensic estimate. It is not "
                "legal proof of document authenticity."
            ),
            styles["BodyText"]
        )
    )

    document.build(
        report_elements
    )

    pdf_buffer.seek(0)

    return pdf_buffer