"""Export, verify, or restore a versioned AgentifyAI curriculum release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from database import SessionLocal  # noqa: E402
from services.content_release_bundle import (  # noqa: E402
    export_content_release,
    restore_content_release,
    verify_content_release,
)


DEFAULT_BUNDLE = ROOT / "data" / "releases" / "ncert_class_11_chemistry_v1.json.gz"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="export approved DB content")
    export.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    export.add_argument("--class-level", default="11")
    export.add_argument("--subject", default="Chemistry")
    export.add_argument("--chapters", nargs="+", type=int, default=list(range(1, 10)))
    export.add_argument("--embedding-model", required=True)
    export.add_argument("--embedding-dimensions", type=int, required=True)
    export.add_argument("--embedding-provider", default="openai-compatible")
    export.add_argument("--embedding-endpoint-host", required=True)

    verify = subparsers.add_parser("verify", help="verify bundle integrity")
    verify.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)

    restore = subparsers.add_parser("restore", help="restore bundle into configured DB")
    restore.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    bundle = args.bundle.resolve()
    if args.command == "verify":
        result = verify_content_release(bundle)
    else:
        db = SessionLocal()
        try:
            if args.command == "export":
                result = export_content_release(
                    db,
                    bundle,
                    class_level=args.class_level,
                    subject=args.subject,
                    chapter_numbers=args.chapters,
                    embedding_model=args.embedding_model,
                    embedding_dimensions=args.embedding_dimensions,
                    embedding_provider=args.embedding_provider,
                    embedding_endpoint_host=args.embedding_endpoint_host,
                )
            else:
                result = restore_content_release(db, bundle)
        finally:
            db.close()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
