"""Restore paper analysis inputs from the pinned Hugging Face dataset revision."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

DATASET = "AlexArutchev/EMDomainTransfer"
REVISION = "de046d0bc22521cf850a2526431b6beb84821934"
ARCHIVE_SHA256 = "7c997030cac727cc3f795294f05add62b35d61cf980ae501e8976948c6bdaf96"


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def safe_target(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset path: {name}")
    target = root / relative
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Dataset path escapes project: {name}")
    return target


def checked_write(root, name, data, expected=None):
    checksum = hashlib.sha256(data).hexdigest()
    if expected and checksum != expected:
        raise ValueError(f"Checksum mismatch: {name}")
    target = safe_target(root, name)
    if target.exists():
        if digest(target) != checksum:
            raise FileExistsError(f"Different local data already exists: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    temporary.write_bytes(data)
    temporary.replace(target)


def restore_archive(archive, root, manifest):
    expected = {r["path"]: r["sha256"] for r in manifest["files"]}
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Unexpected archive member: {member.name}")
            checked_write(root, member.name, source.extractfile(member).read(), expected[member.name])
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive is missing files listed in its manifest")


def restore_parquet(parquet, root, score_files):
    """Project the grouped results table without loading its question/answer columns."""
    import pyarrow.parquet as pq
    allowed = set(score_files)
    for name in allowed:
        safe_target(root, name)
    seen = set()
    current = None
    rows = []

    def flush():
        if current not in allowed:
            return
        if current in seen:
            raise ValueError("Expected results grouped by model and training domain")
        checked_write(root, current, (json.dumps(rows, separators=(",", ":")) + "\n").encode())
        seen.add(current)

    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=4096,
            columns=["model_size", "train_domain", "id", "aligned", "coherent"]):
        for row in batch.to_pylist():
            name = f"output/{row['model_size']}/{row['train_domain']}/scores.json"
            if name != current:
                flush()
                current, rows = name, []
            if name in allowed:
                rows.append({key: row[key] for key in ["id", "aligned", "coherent"]})
    flush()
    if seen != allowed:
        raise ValueError("Results table is missing required model/domain groups")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    cache = args.cache_dir or args.project_root / "data/hf-cache"
    files = {}
    for name in ["artifacts/paper-analysis-manifest.json", "artifacts/paper-analysis.tar.gz",
                 "data/train-00000-of-00001.parquet"]:
        files[name] = Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION, cache_dir=cache))
    manifest = json.loads(files["artifacts/paper-analysis-manifest.json"].read_text())
    archive = files["artifacts/paper-analysis.tar.gz"]
    parquet = files["data/train-00000-of-00001.parquet"]
    if digest(archive) != ARCHIVE_SHA256 or digest(parquet) != manifest["existing_parquet_sha256"]:
        raise ValueError("Downloaded dataset checksum mismatch")
    restore_archive(archive, args.project_root, manifest)
    restore_parquet(parquet, args.project_root, manifest["existing_score_files"])
    print(f"Restored paper analysis inputs under {args.project_root}")


if __name__ == "__main__":
    main()
