import json
import os
import uuid

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for
)

from werkzeug.security import (
    check_password_hash,
    generate_password_hash
)

from werkzeug.utils import secure_filename
from report_generator import generate_analysis_report
from analysis import analyze_document

from database import (
    get_database_connection,
    initialize_database,
    log_activity
)

from signature_analysis import compare_signatures

# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)

app.config["SECRET_KEY"] = (
    "change-this-secret-key-before-deployment"
)


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "pdf"
}

DOCUMENT_TYPES = {
    "pan_card": "PAN Card",
    "aadhaar_card": "Aadhaar Card",
    "marksheet": "Marksheet",
    "educational_certificate": (
        "Educational Certificate"
    ),
    "college_id": "College ID",
    "birth_certificate": "Birth Certificate",
    "income_certificate": "Income Certificate",
    "experience_certificate": (
        "Experience Certificate"
    ),
    "medical_certificate": "Medical Certificate",
    "domicile_certificate": (
        "Domicile Certificate"
    )
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum upload size: 25 MB
app.config["MAX_CONTENT_LENGTH"] = (
    25 * 1024 * 1024
)


# --------------------------------------------------
# Create required folders and database
# --------------------------------------------------

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)

initialize_database()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_extension(filename):
    """
    Return the lowercase extension of a filename.
    """

    if "." not in filename:
        return ""

    return filename.rsplit(
        ".",
        1
    )[1].lower()


def allowed_file(filename):
    """
    Check whether an uploaded file type is supported.
    """

    extension = get_extension(
        filename
    )

    return extension in ALLOWED_EXTENSIONS


# --------------------------------------------------
# User registration
# --------------------------------------------------

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():
    error = None
    success = None

    if request.method == "POST":
        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not name:
            error = "Please enter your name."

        elif not email or "@" not in email:
            error = (
                "Please enter a valid email address."
            )

        elif len(password) < 8:
            error = (
                "Password must contain at least "
                "8 characters."
            )

        elif password != confirm_password:
            error = "Passwords do not match."

        else:
            connection = get_database_connection()

            existing_user = connection.execute(
                """
                SELECT id
                FROM users
                WHERE email = ?
                """,
                (email,)
            ).fetchone()

            if existing_user:
                error = (
                    "An account with this email "
                    "already exists."
                )

            else:
                password_hash = generate_password_hash(
                    password
                )

                connection.execute(
                    """
                    INSERT INTO users (
                        name,
                        email,
                        password_hash
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        name,
                        email,
                        password_hash
                    )
                )

                connection.commit()

                success = (
                    "Registration successful. "
                    "You can now log in."
                )

            connection.close()

    return render_template(
        "register.html",
        error=error,
        success=success
    )


# --------------------------------------------------
# User login
# --------------------------------------------------

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():
    error = None

    if request.method == "POST":
        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        connection = get_database_connection()

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        connection.close()

        if user is None:
            error = "Incorrect email or password."

        elif not check_password_hash(
            user["password_hash"],
            password
        ):
            error = "Incorrect email or password."

        else:
            session.clear()

            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]

            log_activity(
                user["id"],
                "Login",
                "User logged in successfully."
            )

            return redirect(
                url_for("home")
            )

    return render_template(
        "login.html",
        error=error
    )


# --------------------------------------------------
# User logout
# --------------------------------------------------

@app.route("/logout")
def logout():
    user_id = session.get(
        "user_id"
    )

    if user_id:
        log_activity(
            user_id,
            "Logout",
            "User logged out."
        )

    session.clear()

    return redirect(
        url_for("login")
    )


# --------------------------------------------------
# Home and document analysis
# --------------------------------------------------

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():
    error = None

    if request.method == "POST":

        # Require login before document analysis
        if not session.get("user_id"):
            return redirect(
                url_for("login")
            )

        selected_type = request.form.get(
            "document_type",
            ""
        )

        document = request.files.get(
            "document"
        )

        # Validate selected category
        if selected_type not in DOCUMENT_TYPES:
            error = (
                "Please select a valid document type."
            )

        # Validate uploaded file
        elif document is None:
            error = "Please select a document."

        elif document.filename == "":
            error = "Please select a document."

        elif not allowed_file(document.filename):
            error = (
                "Only PNG, JPG, JPEG and PDF "
                "files are supported."
            )

        else:
            original_filename = secure_filename(
                document.filename
            )

            extension = get_extension(
                original_filename
            )

            # Generate a unique stored filename
            unique_filename = (
                f"{uuid.uuid4().hex}.{extension}"
            )

            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                unique_filename
            )

            try:
                # Save uploaded file
                document.save(file_path)

                # Verify category and analyze forgery
                analysis_result = analyze_document(
                    file_path,
                    selected_type
                )

                # Save result to user history
                connection = get_database_connection()

                try:
                    cursor = connection.execute(
                        """
                        INSERT INTO analyses (
                            user_id,
                            original_filename,
                            document_type,
                            document_verified,
                            authenticity_score,
                            risk_score,
                            quality_score,
                            risk_level,
                            forensic_signals,
                            extracted_text
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session["user_id"],
                            original_filename,
                            DOCUMENT_TYPES[selected_type],
                            int(
                                analysis_result[
                                    "type_verified"
                               ]
                            ),
                            analysis_result[
                                "authenticity_score"
                            ],
                            analysis_result[
                                "risk_score"
                            ],
                            analysis_result[
                                "quality_score"
                            ],
                            analysis_result[
                                "risk_level"
                            ],
                            json.dumps(
                                analysis_result.get(
                                    "forensic_signals",
                                    []
                                )
                            ),
                            analysis_result[
                                "extracted_text"
                            ]
                        )
                    )

                    connection.commit()

                    # Obtain the ID before closing the database
                    analysis_id = cursor.lastrowid

                finally:
                    connection.close()

                log_activity(
                    session["user_id"],
                    "Document analysis",
                    (
                        f"Analyzed {original_filename} "
                        f"as {DOCUMENT_TYPES[selected_type]}. "
                        f"Status: {analysis_result['risk_level']}."
                    )
                )

                return render_template(
                    "result.html",
                    filename=original_filename,
                    document_type=DOCUMENT_TYPES[
                        selected_type
                    ],
                    result=analysis_result,
                    analysis_id=analysis_id
                )
            except Exception as exception:
                print(
                    "Document analysis error:",
                    type(exception).__name__,
                    exception
                )

                error = (
                    f"Analysis error: "
                    f"{type(exception).__name__}: "
                    f"{exception}"
                )

    return render_template(
        "index.html",
        error=error
    )

@app.route("/dashboard")
def dashboard():
    if not session.get("user_id"):
        return redirect(
            url_for("login")
        )

    connection = get_database_connection()

    summary = connection.execute(
        """
        SELECT
            COUNT(*) AS total_analyses,

            SUM(
                CASE
                    WHEN document_verified = 1
                    THEN 1
                    ELSE 0
                END
            ) AS verified_documents,

            SUM(
                CASE
                    WHEN document_verified = 0
                    THEN 1
                    ELSE 0
                END
            ) AS rejected_documents,

            ROUND(
                AVG(authenticity_score),
                1
            ) AS average_integrity
        FROM analyses
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()

    risk_rows = connection.execute(
        """
        SELECT
            risk_level,
            COUNT(*) AS total
        FROM analyses
        WHERE user_id = ?
        GROUP BY risk_level
        """,
        (session["user_id"],)
    ).fetchall()

    type_rows = connection.execute(
        """
        SELECT
            document_type,
            COUNT(*) AS total
        FROM analyses
        WHERE user_id = ?
        GROUP BY document_type
        ORDER BY total DESC
        """,
        (session["user_id"],)
    ).fetchall()

    recent_analyses = connection.execute(
        """
        SELECT *
        FROM analyses
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 5
        """,
        (session["user_id"],)
    ).fetchall()

    connection.close()

    risk_labels = [
        row["risk_level"]
        for row in risk_rows
    ]

    risk_values = [
        row["total"]
        for row in risk_rows
    ]

    type_labels = [
        row["document_type"]
        for row in type_rows
    ]

    type_values = [
        row["total"]
        for row in type_rows
    ]

    return render_template(
        "dashboard.html",
        summary=summary,
        risk_labels=risk_labels,
        risk_values=risk_values,
        type_labels=type_labels,
        type_values=type_values,
        recent_analyses=recent_analyses
    )

# --------------------------------------------------
# Analysis history
# --------------------------------------------------

@app.route("/history")
def history():
    if not session.get("user_id"):
        return redirect(
            url_for("login")
        )

    connection = get_database_connection()

    analyses = connection.execute(
        """
        SELECT *
        FROM analyses
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (session["user_id"],)
    ).fetchall()

    connection.close()

    return render_template(
        "history.html",
        analyses=analyses
    )


# --------------------------------------------------
# File-size error
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error=(
            "The selected file is larger than "
            "25 MB. Please upload a smaller file."
        )
    ), 413


# --------------------------------------------------
# Page-not-found error
# --------------------------------------------------

@app.errorhandler(404)
def page_not_found(error):
    return render_template(
        "index.html",
        error="The requested page was not found."
    ), 404


# --------------------------------------------------
# PDF report download
# --------------------------------------------------

@app.route("/report/<int:analysis_id>")
def download_report(analysis_id):
    if not session.get("user_id"):
        return redirect(
            url_for("login")
        )

    connection = get_database_connection()

    analysis = connection.execute(
        """
        SELECT *
        FROM analyses
        WHERE id = ?
        AND user_id = ?
        """,
        (
            analysis_id,
            session["user_id"]
        )
    ).fetchone()

    connection.close()

    if analysis is None:
        return render_template(
            "index.html",
            error="Analysis record not found."
        ), 404

    pdf_report = generate_analysis_report(
        analysis
    )

    log_activity(
        session["user_id"],
        "Report download",
        (
            f"Downloaded report for "
            f"{analysis['original_filename']}."
        )
    )

    return send_file(
        pdf_report,
        as_attachment=True,
        download_name=(
            f"verification-report-{analysis_id}.pdf"
        ),
        mimetype="application/pdf"
    )


# --------------------------------------------------
# File-size error handler
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error=(
            "The selected file is larger than "
            "25 MB. Please upload a smaller file."
        )
    ), 413

@app.route("/activity")
def activity():
    if not session.get("user_id"):
        return redirect(
            url_for("login")
        )

    connection = get_database_connection()

    activity_logs = connection.execute(
        """
        SELECT *
        FROM activity_logs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 100
        """,
        (session["user_id"],)
    ).fetchall()

    connection.close()

    return render_template(
        "activity.html",
        activity_logs=activity_logs
    )

@app.route(
    "/signature-verification",
    methods=["GET", "POST"]
)
def signature_verification():
    if not session.get("user_id"):
        return redirect(
            url_for("login")
        )

    error = None
    result = None

    if request.method == "POST":
        questioned_file = request.files.get(
            "questioned_signature"
        )

        reference_file = request.files.get(
            "reference_signature"
        )

        if (
            questioned_file is None
            or questioned_file.filename == ""
        ):
            error = (
                "Please upload the questioned signature."
            )

        elif (
            reference_file is None
            or reference_file.filename == ""
        ):
            error = (
                "Please upload the genuine reference "
                "signature."
            )

        else:
            questioned_extension = get_extension(
                questioned_file.filename
            )

            reference_extension = get_extension(
                reference_file.filename
            )

            allowed_signature_extensions = {
                "png",
                "jpg",
                "jpeg"
            }

            if (
                questioned_extension
                not in allowed_signature_extensions
                or reference_extension
                not in allowed_signature_extensions
            ):
                error = (
                    "Signature files must be PNG, "
                    "JPG or JPEG images."
                )

            else:
                questioned_name = (
                    f"questioned-{uuid.uuid4().hex}."
                    f"{questioned_extension}"
                )

                reference_name = (
                    f"reference-{uuid.uuid4().hex}."
                    f"{reference_extension}"
                )

                questioned_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    questioned_name
                )

                reference_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    reference_name
                )

                try:
                    questioned_file.save(
                        questioned_path
                    )

                    reference_file.save(
                        reference_path
                    )

                    result = compare_signatures(
                        questioned_path,
                        reference_path
                    )

                    log_activity(
                        session["user_id"],
                        "Signature verification",
                        (
                            "Compared a questioned "
                            "signature with a reference."
                        )
                    )

                except Exception as exception:
                    print(
                        "Signature analysis error:",
                        type(exception).__name__,
                        exception
                    )

                    error = (
                        "The signatures could not be "
                        "analyzed. Upload clear signature "
                        "images on plain backgrounds."
                    )

    return render_template(
        "signature.html",
        error=error,
        result=result
    )
# --------------------------------------------------
# Page-not-found error handler
# --------------------------------------------------

@app.errorhandler(404)
def page_not_found(error):
    return render_template(
        "index.html",
        error="The requested page was not found."
    ), 404


# --------------------------------------------------
# Start Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True
    )