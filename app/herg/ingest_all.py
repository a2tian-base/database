from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Callable

from .config import DbConfig, HttpConfig, IdentifierRunConfig, RunConfig
from .identifier_pipeline import run_identifier_pipeline
from .pipeline import IngestionProgress as SourceIngestionProgress
from .pipeline import run_pipeline
from .sources.chembl import CHEMBL_BASE_URL, ChemblAdapter
from .sources.pubchem import PUBCHEM_BASE_URL, PubChemAdapter
from .sources.unichem_identifiers import (
    DEFAULT_BATCH_SIZE,
    TARGET_SOURCE_IDS,
    UNICHEM_BASE_URL,
    UniChemIdentifierAdapter,
)


ProgressLogger = Callable[[str], None]
ProgressCallback = Callable[["IngestProgress"], None]
SourceProgressCallback = Callable[[SourceIngestionProgress], None]


@dataclass(frozen=True)
class IngestProgress:
    message: str
    step_name: str
    step_index: int
    total_steps: int
    step_fraction: float = 0.0

    @property
    def overall_fraction(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        fraction = (self.step_index + self.step_fraction) / self.total_steps
        return min(1.0, max(0.0, fraction))


@dataclass(frozen=True)
class IngestAllConfig:
    request_timeout_seconds: int = 45
    http_retries: int = 4
    commit_every: int = 500
    fail_fast: bool = False
    max_records: int | None = None


@dataclass(frozen=True)
class IngestStepResult:
    name: str
    kind: str
    success: bool
    stats: dict[str, object]
    error: str = ""


def _emit(progress_logger: ProgressLogger | None, message: str) -> None:
    if progress_logger is not None:
        progress_logger(message)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    message: str,
    step_name: str,
    step_index: int,
    total_steps: int,
    step_fraction: float,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        IngestProgress(
            message=message,
            step_name=step_name,
            step_index=step_index,
            total_steps=total_steps,
            step_fraction=step_fraction,
        )
    )


def _emit_source_progress(
    source_progress_callback: SourceProgressCallback | None,
    *,
    source_name: str,
    phase: str,
) -> None:
    if source_progress_callback is None:
        return
    source_progress_callback(
        SourceIngestionProgress(
            source_name=source_name,
            phase=phase,
            processed=0,
            stored=0,
            skipped_invalid=0,
            failed=0,
            warnings=0,
        )
    )


def _ingestion_success(stats: dict[str, object]) -> bool:
    return int(stats.get("failed") or 0) == 0


def _identifier_success(stats: dict[str, object]) -> bool:
    return int(stats.get("failed") or 0) == 0 and int(stats.get("conflict") or 0) == 0


def _failed_step(name: str, kind: str, exc: Exception) -> IngestStepResult:
    return IngestStepResult(
        name=name,
        kind=kind,
        success=False,
        stats={},
        error=str(exc),
    )


def run_ingest_and_enrich_all(
    *,
    db_config: DbConfig | None = None,
    config: IngestAllConfig | None = None,
    progress_logger: ProgressLogger | None = None,
    progress_callback: ProgressCallback | None = None,
    source_progress_callback: SourceProgressCallback | None = None,
) -> list[IngestStepResult]:
    db_config = db_config or DbConfig.from_env()
    config = config or IngestAllConfig()
    http_config = HttpConfig(
        request_timeout_seconds=config.request_timeout_seconds,
        http_retries=config.http_retries,
    )
    run_config = RunConfig(
        dry_run=False,
        max_records=config.max_records,
        commit_every=config.commit_every,
        fail_fast=config.fail_fast,
    )
    identifier_run_config = IdentifierRunConfig(
        dry_run=False,
        max_records=config.max_records,
        commit_every=config.commit_every,
        fail_fast=config.fail_fast,
    )
    results: list[IngestStepResult] = []
    target_namespaces = list(TARGET_SOURCE_IDS)
    total_steps = 2 + len(target_namespaces)

    step_index = 0
    step_name = "ChEMBL"
    message = "Starting ChEMBL ingestion..."
    _emit(progress_logger, message)
    _emit_source_progress(source_progress_callback, source_name="chembl", phase="starting")
    _emit_progress(
        progress_callback,
        message=message,
        step_name=step_name,
        step_index=step_index,
        total_steps=total_steps,
        step_fraction=0.0,
    )
    try:
        chembl_stats = run_pipeline(
            ChemblAdapter(
                http_config=http_config,
                base_url=CHEMBL_BASE_URL,
                target_chembl_id="CHEMBL240",
                standard_type="IC50",
                relations="=,<,>",
                activity_page_size=1000,
                molecule_batch_size=150,
            ),
            db_config,
            run_config,
            progress_callback=source_progress_callback,
        )
        chembl_payload = asdict(chembl_stats)
        results.append(
            IngestStepResult(
                name="ChEMBL",
                kind="source_ingestion",
                success=_ingestion_success(chembl_payload),
                stats=chembl_payload,
            )
        )
    except Exception as exc:
        results.append(_failed_step("ChEMBL", "source_ingestion", exc))
    message = "Finished ChEMBL ingestion."
    _emit(progress_logger, message)
    _emit_progress(
        progress_callback,
        message=message,
        step_name=step_name,
        step_index=step_index,
        total_steps=total_steps,
        step_fraction=1.0,
    )

    step_index = 1
    step_name = "PubChem"
    message = "Starting PubChem ingestion..."
    _emit(progress_logger, message)
    _emit_source_progress(source_progress_callback, source_name="pubchem", phase="starting")
    _emit_progress(
        progress_callback,
        message=message,
        step_name=step_name,
        step_index=step_index,
        total_steps=total_steps,
        step_fraction=0.0,
    )
    try:
        pubchem_stats = run_pipeline(
            PubChemAdapter(
                http_config=http_config,
                base_url=PUBCHEM_BASE_URL,
                target_gene_symbol="KCNH2",
                target_gene_id="3757",
                activity_name_regex=r"(?i)\bic50\b",
                cid_batch_size=150,
            ),
            db_config,
            run_config,
            progress_callback=source_progress_callback,
        )
        pubchem_payload = asdict(pubchem_stats)
        results.append(
            IngestStepResult(
                name="PubChem",
                kind="source_ingestion",
                success=_ingestion_success(pubchem_payload),
                stats=pubchem_payload,
            )
        )
    except Exception as exc:
        results.append(_failed_step("PubChem", "source_ingestion", exc))
    message = "Finished PubChem ingestion."
    _emit(progress_logger, message)
    _emit_progress(
        progress_callback,
        message=message,
        step_name=step_name,
        step_index=step_index,
        total_steps=total_steps,
        step_fraction=1.0,
    )

    _emit(progress_logger, "Scanning UniChem identifier enrichment candidates...")
    for namespace_index, namespace in enumerate(target_namespaces, start=2):
        step_name = f"UniChem {namespace}"
        message = f"Starting {step_name} enrichment..."
        _emit(progress_logger, message)
        _emit_progress(
            progress_callback,
            message=message,
            step_name=step_name,
            step_index=namespace_index,
            total_steps=total_steps,
            step_fraction=0.0,
        )

        def namespace_progress(message: str) -> None:
            _emit(progress_logger, message)
            match = re.search(r"Progress: (\d+)/(\d+) candidate rows", message)
            if not match:
                return
            processed = int(match.group(1))
            total = int(match.group(2))
            if total <= 0:
                return
            _emit_progress(
                progress_callback,
                message=message,
                step_name=step_name,
                step_index=namespace_index,
                total_steps=total_steps,
                step_fraction=processed / total,
            )

        try:
            adapter = UniChemIdentifierAdapter(
                http_config=http_config,
                target_namespace=namespace,
                base_url=UNICHEM_BASE_URL,
                limit=config.max_records,
                enrich_batch_size=DEFAULT_BATCH_SIZE,
                db_config=db_config,
                progress_logger=namespace_progress,
            )
            candidate_count = adapter.load_candidates()
            stats = run_identifier_pipeline(adapter, db_config, identifier_run_config)
            payload = {
                "candidate_rows_found": candidate_count,
                **asdict(stats),
            }
            results.append(
                IngestStepResult(
                    name=step_name,
                    kind="identifier_enrichment",
                    success=_identifier_success(payload),
                    stats=payload,
                )
            )
        except Exception as exc:
            results.append(_failed_step(step_name, "identifier_enrichment", exc))
        message = f"Finished {step_name} enrichment."
        _emit(progress_logger, message)
        _emit_progress(
            progress_callback,
            message=message,
            step_name=step_name,
            step_index=namespace_index,
            total_steps=total_steps,
            step_fraction=1.0,
        )

    _emit(progress_logger, "Ingestion and enrichment run complete.")
    _emit_progress(
        progress_callback,
        message="Ingestion and enrichment run complete.",
        step_name="Complete",
        step_index=total_steps,
        total_steps=total_steps,
        step_fraction=0.0,
    )
    return results
