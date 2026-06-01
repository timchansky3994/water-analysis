"""Readiness status badge component."""

from __future__ import annotations

import streamlit as st

from water_analysis.profiling.readiness import ReadinessAssessment
from water_analysis.reporting.specialist_summary import READINESS_REASON_RU, READINESS_STATUS_RU


def render_readiness_badge(assessment: ReadinessAssessment) -> None:
    """Render a coloured readiness status badge with issue explanations."""
    status = assessment.status
    description = READINESS_STATUS_RU.get(status, status)
    label = f"Пригодность данных: **{status}** — {description}"

    if status == "suitable":
        st.success(label)
    elif status == "weakly_suitable":
        st.warning(label)
    else:
        st.error(label)

    if assessment.issues:
        with st.expander("Причины и ограничения"):
            for issue in assessment.issues:
                explanation = READINESS_REASON_RU.get(issue.code, issue.message)
                severity_label = "критично" if issue.severity == "critical" else "предупреждение"
                st.caption(f"**{severity_label}**: {explanation} (`{issue.code}`)")
