"""
backend/tools/report_tool.py
Generates a downloadable PDF report using ReportLab.
"""
import logging
import io
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

from config import REPORTS_DIR
from .stats_tool import (
    get_missing_values, get_correlations, get_outliers,
    compute_data_quality_score, generate_automatic_insights,
)

logger = logging.getLogger(__name__)

# ── Colour scheme ──────────────────────────────────────────────────────────
PRIMARY   = colors.HexColor("#1a1a2e")
ACCENT    = colors.HexColor("#e94560")
LIGHT     = colors.HexColor("#f0f0f0")
MID       = colors.HexColor("#cccccc")
WHITE     = colors.white
TEXT      = colors.HexColor("#2d2d2d")


def _styles():
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontSize=22, textColor=PRIMARY,
                                spaceAfter=6, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                              fontSize=14, textColor=PRIMARY,
                              spaceBefore=14, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                              fontSize=11, textColor=ACCENT,
                              spaceBefore=10, spaceAfter=2),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontSize=9, textColor=TEXT,
                               leading=14, spaceAfter=4),
        "small": ParagraphStyle("small", parent=base["Normal"],
                                fontSize=8, textColor=colors.grey,
                                leading=12),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"],
                                 fontSize=9, textColor=TEXT,
                                 leftIndent=12, leading=14,
                                 bulletIndent=4),
    }
    return custom


def _section_header(title: str, styles: dict):
    return [
        Spacer(1, 0.3 * cm),
        Paragraph(title, styles["h1"]),
        HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=4),
    ]


def _kv_table(rows: list[tuple], col_widths=None):
    """Two-column key-value table."""
    col_widths = col_widths or [7 * cm, 10 * cm]
    tbl = Table(rows, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, MID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def _data_table(headers: list, rows: list, zebra: bool = True):
    """Generic data table with optional zebra striping."""
    all_rows = [headers] + rows
    tbl = Table(all_rows, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, MID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if zebra:
        for i, _ in enumerate(rows):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i + 1), (-1, i + 1), LIGHT))
    tbl.setStyle(TableStyle(style))
    return tbl


def generate_pdf_report(
    dataset_name: str,
    metadata: dict,
    chart_paths: list[str] = None,
    ai_summary: str = "",
) -> str:
    """
    Generate a full PDF report and save to REPORTS_DIR.

    Args:
        dataset_name: Human-readable dataset name.
        metadata: Extracted metadata dict.
        chart_paths: Optional list of saved chart PNG paths to embed.
        ai_summary: AI-generated executive summary text.

    Returns:
        Path to the generated PDF file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORTS_DIR / f"NulAware_Report_{dataset_name}_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"NulAware AI Report – {dataset_name}",
        author="NulAware AI",
    )

    styles = _styles()
    story = []
    ds = metadata.get("dataset_summary", {})
    quality = compute_data_quality_score(metadata)
    insights = generate_automatic_insights(metadata)

    # ── Cover ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("NulAware AI", styles["h2"]))
    story.append(Paragraph(f"Data Profiling Report", styles["title"]))
    story.append(Paragraph(f"Dataset: <b>{dataset_name}</b>", styles["body"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
        styles["small"],
    ))
    story.append(Spacer(1, 0.5 * cm))

    # Quality score badge
    grade_color = {"A": "#27ae60", "B": "#2980b9", "C": "#e67e22", "D": "#e74c3c"}
    story.append(_kv_table([
        ("Data Quality Score", f"{quality['total_score']} / 100  (Grade {quality['grade']})"),
        ("Total Rows", str(ds.get("n_rows", "N/A"))),
        ("Total Columns", str(ds.get("n_columns", "N/A"))),
        ("Missing Cells", f"{ds.get('n_missing_cells', 0)} ({round(ds.get('pct_missing_cells', 0)*100, 2)}%)"),
        ("Duplicate Rows", f"{ds.get('n_duplicate_rows', 0)} ({round(ds.get('pct_duplicate_rows', 0)*100, 2)}%)"),
    ]))
    story.append(PageBreak())

    # ── 1. Executive Summary ────────────────────────────────────────────────
    story += _section_header("1. Executive Summary", styles)
    if ai_summary:
        for para in ai_summary.split("\n\n"):
            story.append(Paragraph(para.strip(), styles["body"]))
    else:
        story.append(Paragraph(
            f"This report provides a comprehensive profiling of the <b>{dataset_name}</b> dataset "
            f"containing {ds.get('n_rows', '?')} rows and {ds.get('n_columns', '?')} columns. "
            "Key findings are summarised below.",
            styles["body"],
        ))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>Automatic Insights:</b>", styles["body"]))
    for insight in insights:
        story.append(Paragraph(f"• {insight}", styles["bullet"]))

    # ── 2. Dataset Information ───────────────────────────────────────────────
    story += _section_header("2. Dataset Information", styles)
    col_types = ds.get("types", {})
    type_str = ", ".join(f"{k}: {v}" for k, v in col_types.items()) if col_types else "N/A"
    story.append(_kv_table([
        ("Rows", str(ds.get("n_rows", "N/A"))),
        ("Columns", str(ds.get("n_columns", "N/A"))),
        ("Column Types", type_str),
        ("Memory Size", f"{ds.get('memory_size', 0):,} bytes"),
    ]))

    # ── 3. Missing Values Analysis ──────────────────────────────────────────
    story += _section_header("3. Missing Values Analysis", styles)
    mv = get_missing_values(metadata)
    mv_with_missing = {k: v for k, v in mv.items() if v["n_missing"] > 0}
    if mv_with_missing:
        headers = ["Column", "Missing Count", "Missing %", "Status"]
        rows = []
        for col, v in list(mv_with_missing.items())[:20]:
            status = "Critical" if v["p_missing"] > 20 else "High" if v["p_missing"] > 5 else "Low"
            rows.append([col, str(v["n_missing"]), f"{v['p_missing']}%", status])
        story.append(_data_table(headers, rows))
    else:
        story.append(Paragraph("✓ No missing values detected.", styles["body"]))

    # ── 4. Correlation Analysis ──────────────────────────────────────────────
    story += _section_header("4. Correlation Analysis", styles)
    corrs = get_correlations(metadata, threshold=0.5)
    if corrs:
        headers = ["Column A", "Column B", "Pearson r", "Strength"]
        rows = []
        for p in corrs[:15]:
            r = abs(p["pearson"])
            strength = "Very Strong" if r >= 0.9 else "Strong" if r >= 0.7 else "Moderate"
            rows.append([p["col_a"], p["col_b"], str(p["pearson"]), strength])
        story.append(_data_table(headers, rows))
    else:
        story.append(Paragraph("No strong correlations found (|r| ≥ 0.5).", styles["body"]))

    # ── 5. Outlier Analysis ──────────────────────────────────────────────────
    story += _section_header("5. Outlier Analysis (IQR Method)", styles)
    outliers = get_outliers(metadata)
    if outliers:
        headers = ["Column", "IQR", "Lower Bound", "Upper Bound", "Min", "Max"]
        rows = []
        for col, v in list(outliers.items())[:15]:
            rows.append([
                col,
                f"{v.get('iqr', 'N/A'):.2f}" if v.get("iqr") else "N/A",
                f"{v.get('lower_bound', 'N/A'):.2f}" if v.get("lower_bound") is not None else "N/A",
                f"{v.get('upper_bound', 'N/A'):.2f}" if v.get("upper_bound") is not None else "N/A",
                f"{v.get('min', 'N/A'):.2f}" if v.get("min") is not None else "N/A",
                f"{v.get('max', 'N/A'):.2f}" if v.get("max") is not None else "N/A",
            ])
        story.append(_data_table(headers, rows))
    else:
        story.append(Paragraph("✓ No outliers detected using IQR method.", styles["body"]))

    # ── 6. Duplicate Detection ───────────────────────────────────────────────
    story += _section_header("6. Duplicate Detection", styles)
    dups = metadata.get("duplicates", {})
    story.append(_kv_table([
        ("Duplicate Rows", str(dups.get("n_duplicate_rows", 0))),
        ("Percentage", f"{round(dups.get('pct_duplicate_rows', 0)*100, 2)}%"),
        ("Recommendation",
         "Remove duplicate rows before analysis." if dups.get("n_duplicate_rows", 0) > 0
         else "Dataset is duplicate-free."),
    ]))

    # ── 7. Column Statistics ─────────────────────────────────────────────────
    story += _section_header("7. Column Statistics", styles)
    numeric_types = ("Numeric", "Real", "Integer", "Float")
    num_vars = {
        col: v for col, v in metadata.get("variables", {}).items()
        if v.get("type") in numeric_types
    }
    if num_vars:
        headers = ["Column", "Mean", "Median", "Std", "Min", "Max", "Skew"]

        def _f(val):
            if val is None:
                return "N/A"
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val)

        rows = [
            [col, _f(v.get("mean")), _f(v.get("median")), _f(v.get("std")),
             _f(v.get("min")), _f(v.get("max")), _f(v.get("skewness"))]
            for col, v in list(num_vars.items())[:20]
        ]
        story.append(_data_table(headers, rows))
    else:
        story.append(Paragraph("No numeric columns found.", styles["body"]))

    # ── 8. Charts ────────────────────────────────────────────────────────────
    if chart_paths:
        story.append(PageBreak())
        story += _section_header("8. Charts", styles)
        for chart_path in chart_paths:
            p = Path(chart_path)
            if p.exists():
                try:
                    img = RLImage(str(p), width=16 * cm, height=9 * cm,
                                  kind="proportional")
                    story.append(img)
                    story.append(Paragraph(p.stem.replace("_", " ").title(),
                                           styles["small"]))
                    story.append(Spacer(1, 0.3 * cm))
                except Exception as e:
                    logger.warning(f"Could not embed chart {p.name}: {e}")

    # ── 9. AI Recommendations ────────────────────────────────────────────────
    story += _section_header("9. AI Recommendations", styles)
    recs = _build_recommendations(metadata, quality)
    for rec in recs:
        story.append(Paragraph(f"• {rec}", styles["bullet"]))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
    story.append(Paragraph(
        "Generated by <b>NulAware AI</b> — Intelligent Data Profiling Assistant",
        styles["small"],
    ))

    doc.build(story)
    logger.info(f"PDF report saved to {pdf_path}")
    return str(pdf_path)


def _build_recommendations(metadata: dict, quality: dict) -> list[str]:
    recs = []
    ds = metadata.get("dataset_summary", {})

    # Missing values
    mv = {col: v for col, v in metadata.get("variables", {}).items()
          if v.get("p_missing", 0) > 0.1}
    if mv:
        recs.append(
            f"Handle missing values in {len(mv)} column(s). "
            "Consider imputation (mean/median for numeric, mode for categorical) or removal."
        )

    # Duplicates
    if ds.get("n_duplicate_rows", 0) > 0:
        recs.append(
            f"Remove {ds['n_duplicate_rows']} duplicate rows to avoid biased analysis."
        )

    # Outliers
    outliers = {col: v for col, v in metadata.get("outliers", {}).items()
                if v.get("has_outliers")}
    if outliers:
        recs.append(
            f"Investigate outliers in {len(outliers)} column(s). "
            "Consider capping, transformation, or domain-specific treatment."
        )

    # Skewness
    for col, v in metadata.get("variables", {}).items():
        skew = v.get("skewness")
        if skew and abs(skew) > 1:
            recs.append(
                f"Apply log or Box-Cox transformation on '{col}' to reduce skewness ({skew:.2f})."
            )
            break

    # Strong correlations
    corrs = metadata.get("correlations", {}).get("pearson_pairs", [])
    very_strong = [p for p in corrs if abs(p.get("pearson", 0)) > 0.95]
    if very_strong:
        recs.append(
            f"Consider removing one of highly collinear features: "
            f"'{very_strong[0]['col_a']}' & '{very_strong[0]['col_b']}' (r={very_strong[0]['pearson']})."
        )

    if quality["total_score"] >= 85:
        recs.append("Dataset is in good shape. Proceed to modelling with standard preprocessing.")
    elif quality["total_score"] >= 60:
        recs.append("Dataset requires moderate cleaning. Address missing values and outliers before modelling.")
    else:
        recs.append("Dataset requires significant cleaning. Prioritise missing value treatment and deduplication.")

    return recs