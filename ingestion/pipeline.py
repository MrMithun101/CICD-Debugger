import json
import logging
from pathlib import Path

from ingestion.cleaner import clean_log
from ingestion.chunker import chunk_log
from ingestion.classifier import classify_failure
from ingestion.models import ProcessedChunk, ProcessedLog

logger = logging.getLogger(__name__)


def process_log(
    raw_text: str,
    meta: dict,
    processed_dir: Path,
) -> ProcessedLog:
    """Transform a raw log + its metadata dict into a ProcessedLog and save it.

    Steps:
      1. Clean — strip timestamps, ANSI codes, GHA commands
      2. Classify — label the whole log with a failure type
      3. Chunk — split into per-step sections
      4. Build ProcessedLog + save JSON to processed_dir
    """
    cleaned = clean_log(raw_text)
    failure_type = classify_failure(cleaned)
    raw_chunks = chunk_log(cleaned)

    repo_safe = meta["repo"].replace("/", "__")
    run_id = meta["run_id"]

    chunks = [
        ProcessedChunk(
            chunk_id=f"{repo_safe}__{run_id}__chunk_{i}",
            chunk_index=i,
            source_step=c["step"],
            text=c["text"],
            repo=meta["repo"],
            workflow_name=meta["workflow_name"],
            run_id=run_id,
            failure_type=failure_type,
            created_at=meta["created_at"],
            html_url=meta["html_url"],
        )
        for i, c in enumerate(raw_chunks)
    ]

    log = ProcessedLog(
        repo=meta["repo"],
        workflow_name=meta["workflow_name"],
        run_id=run_id,
        failure_type=failure_type,
        created_at=meta["created_at"],
        html_url=meta["html_url"],
        chunks=chunks,
    )

    _save(log, processed_dir)
    logger.info(
        "Processed run %d (%s) → %d chunks, type=%s",
        run_id, meta["repo"], len(chunks), failure_type,
    )
    return log


def _save(log: ProcessedLog, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    repo_safe = log.repo.replace("/", "__")
    out_path = processed_dir / f"{repo_safe}__{log.run_id}.json"

    payload = {
        "repo": log.repo,
        "workflow_name": log.workflow_name,
        "run_id": log.run_id,
        "failure_type": log.failure_type,
        "created_at": log.created_at,
        "html_url": log.html_url,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "source_step": c.source_step,
                "text": c.text,
            }
            for c in log.chunks
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(data_dir: Path) -> list[ProcessedLog]:
    """Process all raw logs found in data_dir/raw_logs/ and save to data_dir/processed/."""
    raw_logs_dir = data_dir / "raw_logs"
    processed_dir = data_dir / "processed"

    meta_files = list(raw_logs_dir.glob("*.json"))
    if not meta_files:
        logger.warning("No metadata files found in %s", raw_logs_dir)
        return []

    results: list[ProcessedLog] = []
    for meta_path in meta_files:
        log_path = meta_path.with_suffix(".txt")
        if not log_path.exists():
            logger.warning("Log file missing for %s — skipping", meta_path.name)
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw_text = log_path.read_text(encoding="utf-8")

        try:
            processed = process_log(raw_text, meta, processed_dir)
            results.append(processed)
        except Exception as e:
            logger.error("Failed to process %s: %s", meta_path.name, e)

    return results
