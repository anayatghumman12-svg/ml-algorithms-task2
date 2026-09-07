"""
generate_report.py
-------------------
Ye script results/ folder ke numbers aur images utha kar ek simple,
non-technical stakeholder-facing PDF report banata hai (reports/final_report.pdf).

Isko main.py ke baad ek dafa chalao:
    python generate_report.py
"""

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)

from src import settings

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20)
heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], spaceBefore=14)
body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=11, leading=16)


def build_report():
    """Reads results and builds the final stakeholder PDF report."""

    df = pd.read_csv(settings.RESULTS_CSV_PATH)
    best_row = df.sort_values(by=settings.PRIMARY_METRIC, ascending=False).iloc[0]

    doc = SimpleDocTemplate(settings.REPORT_PATH, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story = []

    # ---- Title ----
    story.append(Paragraph("Customer Churn Prediction — Final Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "NextBridge Summer Internship 2026 — AI/ML Track", body_style
    ))
    story.append(Spacer(1, 20))

    # ---- Summary for non-technical reader ----
    story.append(Paragraph("What was the goal?", heading_style))
    story.append(Paragraph(
        "We built a machine learning model that predicts which telecom customers "
        "are likely to cancel their service (churn). This lets the business reach "
        "out to at-risk customers before they leave.", body_style
    ))

    story.append(Paragraph("Which model do we recommend?", heading_style))
    story.append(Paragraph(
        f"We recommend <b>{best_row['model']}</b>, which correctly catches "
        f"<b>{best_row['recall']*100:.1f}%</b> of customers who actually churn "
        f"(this is our chosen priority metric, called recall, because missing a "
        f"churner costs more than a false alarm). Overall accuracy is "
        f"{best_row['accuracy']*100:.1f}%.", body_style
    ))

    # ---- Results table ----
    story.append(Paragraph("Model Comparison", heading_style))
    table_data = [["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
    for _, row in df.iterrows():
        table_data.append([
            row["model"],
            f"{row['accuracy']:.3f}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1_score']:.3f}",
            f"{row['roc_auc']:.3f}",
        ])

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a67")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # ---- Confusion matrix image ----
    story.append(Paragraph("Where does the model make mistakes?", heading_style))
    story.append(Paragraph(
        "The chart below shows correct vs incorrect predictions on unseen "
        "customers (the test set).", body_style
    ))
    story.append(Spacer(1, 8))
    story.append(Image(settings.CONFUSION_MATRIX_PATH, width=4 * inch, height=4 * inch))
    story.append(Spacer(1, 20))

    # ---- Feature importance image ----
    story.append(Paragraph("What drives churn the most?", heading_style))
    story.append(Paragraph(
        "The chart below ranks the customer attributes that influenced the "
        "model's predictions the most.", body_style
    ))
    story.append(Spacer(1, 8))
    story.append(Image(settings.FEATURE_IMPORTANCE_PATH, width=5.5 * inch, height=4.1 * inch))
    story.append(Spacer(1, 20))

    # ---- Conclusion ----
    story.append(Paragraph("Bottom line", heading_style))
    story.append(Paragraph(
        "The model gives the business an early-warning signal for churn risk, "
        "so retention offers can be targeted at the customers most likely to leave, "
        "instead of guessing.", body_style
    ))

    doc.build(story)
    print(f"Report saved to {settings.REPORT_PATH}")


if __name__ == "__main__":
    build_report()
