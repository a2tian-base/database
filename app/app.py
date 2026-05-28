import io
import math
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st

from herg.ingest_all import (
    IngestProgress,
    IngestStepResult,
    SourceIngestionProgress,
    run_ingest_and_enrich_all,
)
from herg.read_db import (
    fetch_dashboard_data,
    fetch_dashboard_metrics,
    fetch_results,
    fetch_results_count,
)


SOURCE_DISPLAY_NAMES = {
    "chembl": "ChEMBL",
    "pubchem": "PubChem",
}


st.set_page_config(page_title="hERG IC50 Database", layout="wide")


def build_histogram_counts(series: pd.Series, bins: int) -> pd.DataFrame:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return pd.DataFrame(columns=["bin_start", "bin_end", "bin", "count"])

    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5

    bucketed = pd.cut(numeric, bins=bins, include_lowest=True)
    counts = bucketed.value_counts(sort=False)
    return pd.DataFrame(
        {
            "bin_start": [float(interval.left) for interval in counts.index],
            "bin_end": [float(interval.right) for interval in counts.index],
            "bin": [f"{interval.left:.2f} to {interval.right:.2f}" for interval in counts.index],
            "count": counts.values,
        }
    )


def _summary_value(stats: dict[str, object], key: str) -> object:
    value = stats.get(key)
    return "" if value is None else value


def build_ingest_summary(results: list[IngestStepResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        stats = result.stats
        rows.append(
            {
                "step": result.name,
                "type": result.kind,
                "status": "complete" if result.success else "needs review",
                "candidates": _summary_value(stats, "candidate_rows_found"),
                "processed": _summary_value(stats, "processed"),
                "stored": _summary_value(stats, "stored"),
                "attached": _summary_value(stats, "attached"),
                "already_present": _summary_value(stats, "already_present"),
                "unmatched": _summary_value(stats, "unmatched"),
                "conflicts": _summary_value(stats, "conflict"),
                "skipped": _summary_value(stats, "skipped_invalid"),
                "failed": _summary_value(stats, "failed"),
                "duration_seconds": _summary_value(stats, "duration_seconds"),
                "error": result.error,
            }
        )
    return pd.DataFrame(rows)


def build_source_progress_summary(source_progress: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for source_name, display_name in SOURCE_DISPLAY_NAMES.items():
        stats = source_progress.get(source_name, {})
        rows.append(
            {
                "source": display_name,
                "phase": stats.get("phase", "waiting"),
                "processed": stats.get("processed", 0),
                "ingested": stats.get("stored", 0),
                "skipped": stats.get("skipped_invalid", 0),
                "failed": stats.get("failed", 0),
                "warnings": stats.get("warnings", 0),
            }
        )
    return pd.DataFrame(rows)


def render_ingest_results(results: list[IngestStepResult], run_log: str) -> None:
    if not results:
        return

    if all(result.success for result in results):
        st.success("Ingest and enrichment completed.")
    else:
        st.warning("Ingest and enrichment completed with issues.")

    st.dataframe(build_ingest_summary(results), use_container_width=True, hide_index=True)

    if run_log.strip():
        with st.expander("Run log"):
            st.code(run_log.strip(), language="text")


st.title("hERG IC50 Database")
st.write("Run ingestion, browse loaded hERG IC50 results, and explore database metrics.")

ingest_tab, results_tab, dashboard_tab = st.tabs(["Ingest", "Browse Results", "Dashboard"])

with ingest_tab:
    st.subheader("Ingest")
    if st.button("Ingest and Enrich All", type="primary", key="ingest_all_btn"):
        progress_lines: list[str] = []
        progress_placeholder = st.empty()
        progress_bar = st.progress(0, text="Starting ingestion...")
        progress_state = {"percent": 0}
        source_progress_state: dict[str, dict[str, object]] = {}
        source_progress_placeholder = st.empty()
        source_progress_placeholder.dataframe(
            build_source_progress_summary(source_progress_state),
            use_container_width=True,
            hide_index=True,
        )

        def log_progress(message: str) -> None:
            progress_lines.append(message)
            progress_placeholder.info(message)

        def update_progress(progress: IngestProgress) -> None:
            percent = int(progress.overall_fraction * 100)
            progress_state["percent"] = percent
            progress_bar.progress(percent, text=f"{percent}% - {progress.message}")

        def update_source_progress(progress: SourceIngestionProgress) -> None:
            source_progress_state[progress.source_name] = {
                "phase": progress.phase,
                "processed": progress.processed,
                "stored": progress.stored,
                "skipped_invalid": progress.skipped_invalid,
                "failed": progress.failed,
                "warnings": progress.warnings,
            }
            source_progress_placeholder.dataframe(
                build_source_progress_summary(source_progress_state),
                use_container_width=True,
                hide_index=True,
            )

        stdout_buffer = io.StringIO()
        try:
            with st.spinner("Running ingestion and enrichment..."), redirect_stdout(stdout_buffer):
                ingest_results = run_ingest_and_enrich_all(
                    progress_logger=log_progress,
                    progress_callback=update_progress,
                    source_progress_callback=update_source_progress,
                )
        except Exception as exc:
            progress_bar.progress(
                progress_state["percent"],
                text=f"{progress_state['percent']}% - Ingestion stopped with an error.",
            )
            ingest_results = [
                IngestStepResult(
                    name="Ingest and Enrich All",
                    kind="workflow",
                    success=False,
                    stats={},
                    error=str(exc),
                )
            ]

        stdout_log = stdout_buffer.getvalue().strip()
        run_log = "\n".join(progress_lines)
        if stdout_log:
            run_log = f"{run_log}\n{stdout_log}" if run_log else stdout_log

        st.session_state["ingest_results"] = ingest_results
        st.session_state["ingest_run_log"] = run_log
        progress_placeholder.empty()

    render_ingest_results(
        st.session_state.get("ingest_results", []),
        st.session_state.get("ingest_run_log", ""),
    )

with results_tab:
    st.subheader("Browse Results")
    limit = st.slider("Rows to preview", min_value=10, max_value=1000, value=100, step=10)
    try:
        total_results = fetch_results_count()
        results_df = fetch_results(limit)
        if total_results == 0 or results_df.empty:
            st.info("No IC50 results found yet.")
        else:
            st.caption(
                f"Previewing {len(results_df):,} of {total_results:,} rows. "
                "The on-screen table stays capped for performance."
            )
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download preview CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="ic50_results_preview.csv",
                mime="text/csv",
            )

            prepare_full_export = st.checkbox(
                "Prepare full CSV export",
                help="Load every row from the database and make it available as a CSV download.",
            )
            if prepare_full_export:
                with st.spinner("Preparing full results export..."):
                    full_results_df = fetch_results(limit=None)
                st.caption(f"Full export contains {len(full_results_df):,} rows.")
                st.download_button(
                    label="Download full results CSV",
                    data=full_results_df.to_csv(index=False).encode("utf-8"),
                    file_name="ic50_results_all.csv",
                    mime="text/csv",
                )
    except Exception as exc:
        st.error(f"Failed to load results: {exc}")

with dashboard_tab:
    st.subheader("Data Dashboard")
    st.write("Summary metrics and distribution views for loaded IC50 records.")

    try:
        metrics = fetch_dashboard_metrics()
    except Exception as exc:
        st.error(f"Failed to load dashboard metrics: {exc}")
        metrics = None

    if metrics is not None:
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        metric_col1.metric("Compounds", f"{metrics['compounds_n']:,}")
        metric_col2.metric("IC50 Entries", f"{metrics['results_n']:,}")
        metric_col3.metric("Compounds With Results", f"{metrics['compounds_with_results_n']:,}")
        metric_col4.metric(
            "First Entry",
            metrics["first_result_at"].strftime("%Y-%m-%d") if metrics["first_result_at"] else "-",
        )
        metric_col5.metric(
            "Latest Entry",
            metrics["last_result_at"].strftime("%Y-%m-%d") if metrics["last_result_at"] else "-",
        )

    max_rows = st.slider(
        "Rows to analyze",
        min_value=100,
        max_value=250000,
        value=20000,
        step=100,
        help="Recent rows to pull from the database for visualization.",
    )

    try:
        dashboard_df = fetch_dashboard_data(limit=max_rows)
    except Exception as exc:
        st.error(f"Failed to load dashboard data: {exc}")
        dashboard_df = pd.DataFrame()

    if dashboard_df.empty:
        st.info("No IC50 data available yet.")
    else:
        dashboard_df["created_at"] = pd.to_datetime(dashboard_df["created_at"], errors="coerce")
        dashboard_df["ic50_um"] = pd.to_numeric(dashboard_df["ic50_um"], errors="coerce")
        dashboard_df["pic50"] = pd.to_numeric(dashboard_df["pic50"], errors="coerce")
        dashboard_df["log10_ic50_um"] = dashboard_df["ic50_um"].apply(
            lambda value: math.log10(value) if pd.notna(value) and value > 0 else None
        )

        st.markdown("### Filters")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            qualifier_options = sorted(dashboard_df["qualifier"].dropna().astype(str).unique().tolist())
            selected_qualifiers = st.multiselect(
                "Qualifier",
                options=qualifier_options,
                default=qualifier_options,
            )
        with filter_col2:
            unit_options = sorted(dashboard_df["ic50_unit"].dropna().astype(str).unique().tolist())
            selected_units = st.multiselect(
                "Unit",
                options=unit_options,
                default=unit_options,
            )

        filtered_df = dashboard_df.copy()
        if selected_qualifiers:
            filtered_df = filtered_df[filtered_df["qualifier"].isin(selected_qualifiers)]
        if selected_units:
            filtered_df = filtered_df[filtered_df["ic50_unit"].isin(selected_units)]

        st.caption(f"Visualizing {len(filtered_df):,} rows after filters.")

        if filtered_df.empty:
            st.info("No data left after filtering.")
        else:
            st.markdown("### Category Distributions")
            dist_col1, dist_col2 = st.columns(2)
            with dist_col1:
                st.caption("Qualifier counts")
                qualifier_counts = (
                    filtered_df["qualifier"]
                    .astype(str)
                    .value_counts()
                    .reindex(["=", "<", ">"], fill_value=0)
                    .reset_index()
                )
                qualifier_counts.columns = ["qualifier", "count"]
                st.bar_chart(qualifier_counts.set_index("qualifier"))
            with dist_col2:
                st.caption("Unit counts")
                unit_counts = (
                    filtered_df["ic50_unit"]
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .reset_index()
                )
                unit_counts.columns = ["unit", "count"]
                st.bar_chart(unit_counts.set_index("unit"))

            st.markdown("### Value Distributions")
            value_col1, value_col2 = st.columns(2)
            with value_col1:
                st.caption("pIC50 histogram")
                pic50_hist = build_histogram_counts(filtered_df["pic50"], bins=30)
                if pic50_hist.empty:
                    st.info("No valid pIC50 values.")
                else:
                    st.bar_chart(pic50_hist.set_index("bin_start")[["count"]])
            with value_col2:
                st.caption("log10(IC50 uM) histogram")
                ic50_log_hist = build_histogram_counts(filtered_df["log10_ic50_um"], bins=30)
                if ic50_log_hist.empty:
                    st.info("No valid IC50 values.")
                else:
                    st.bar_chart(ic50_log_hist.set_index("bin_start")[["count"]])

            st.markdown("### Trend and Top Compounds")
            trend_col1, trend_col2 = st.columns(2)
            with trend_col1:
                st.caption("Entries per month")
                monthly_counts = (
                    filtered_df.dropna(subset=["created_at"])
                    .assign(month=lambda df: df["created_at"].dt.to_period("M").astype(str))
                    .groupby("month")
                    .size()
                    .reset_index(name="count")
                    .sort_values("month")
                )
                if monthly_counts.empty:
                    st.info("No valid timestamps for trend plot.")
                else:
                    st.line_chart(monthly_counts.set_index("month"))
            with trend_col2:
                st.caption("Top compounds by entry count")
                top_compounds = (
                    filtered_df["compound_label"]
                    .astype(str)
                    .value_counts()
                    .head(15)
                    .reset_index()
                )
                top_compounds.columns = ["compound", "count"]
                st.bar_chart(top_compounds.set_index("compound"))
