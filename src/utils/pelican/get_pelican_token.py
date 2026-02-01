import argparse
import asyncio
import logging
import sys
from pathlib import Path
from pelicanfs import PelicanFileSystem

try:
    # When run from repo root: `python utils/pelican/get_pelican_token.py ...`
    from utils.pelican.token_lib import (
        get_access_token_async,
        normalize_scope_path,
    )
except Exception:
    # Fallback: allow running from other working directories.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.pelican.token_lib import (
        get_access_token_async,
        normalize_scope_path,
    )


async def pelican(
    source_path: str,
    target_path: str,
    data: str,
    auth_cache_file: str = ".pelican_auth_cache",
    oidc_url: str = "https://token-issuer.icecube.aq",
    federation: str = "osdf://",
    storage_prefix: str = "/icecube/wipac",
    ls_source: bool = False,
    ls_max_entries: int = 50,
) -> str:
    source_path = normalize_scope_path(source_path)
    target_path = normalize_scope_path(target_path)
    storage_prefix = storage_prefix.rstrip("/")
    full_path = f"{storage_prefix}{target_path}"

    access_token = await get_access_token_async(
        oidc_url=oidc_url,
        source_path=source_path,
        target_path=target_path,
        auth_cache_file=auth_cache_file,
        want_modify=True,
    )

    print(federation, full_path)

    pelfs = PelicanFileSystem(
        federation,
        headers={"Authorization": f"Bearer {access_token}"},
        direct_reads=True,
    )

    if ls_source:
        def _print_ls(path: str) -> bool:
            try:
                entries = pelfs.ls(path)
            except Exception as e:
                print(f"ls failed for {path}: {e}")
                return False

            try:
                n_entries = len(entries)
            except Exception:
                n_entries = None

            print(f"ls {path} -> {n_entries if n_entries is not None else '?'} entries")
            for item in entries[: max(0, int(ls_max_entries))]:
                print(item)
            if n_entries is not None and n_entries > int(ls_max_entries):
                print(f"... ({n_entries - int(ls_max_entries)} more)")
            return True

        # First try exactly what the user requested.
        ok = _print_ls(source_path)
        # If that fails and doesn't already include the storage prefix, try prefixing.
        if not ok and not source_path.startswith(storage_prefix):
            _print_ls(f"{storage_prefix}{source_path}")


    print(f"Writing data to Pelican path: {federation}{full_path}")

    # fsspec currently does not support writing files with over HTTPFileSystem
    # using 'with fs.open(...) as f' syntax, so we need to generate the file 
    # locally and then upload/put it.

    with open("temp_pelican_token.txt", "w") as f:
        f.write(data)
    pelfs.put("temp_pelican_token.txt", f"{full_path}/test.token.pelicanfs")
    # Path("temp_pelican_token.txt").unlink()

    print(f"Successfully wrote data to Pelican path: {federation}{full_path}/test.token.pelicanfs")
    return access_token


def main(argv: list[str] | None = None) -> int:
    # Ensure device-flow instructions are visible even if the caller doesn't configure logging.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Fetch a Pelican access token (device flow) and optionally write data to a Pelican path.",
    )
    parser.add_argument(
        "--source-path",
        required=True,
        help="Path used to request read scope, e.g. /icecube/wipac/...",
    )
    parser.add_argument(
        "--target-path",
        required=False,
        default=None,
        help=(
            "Path used to request modify scope and (by default) where data will be written. "
            "Optional in token-only mode; defaults to --source-path."
        ),
    )
    parser.add_argument(
        "--auth-cache-file",
        default=".pelican_auth_cache",
        help="Path to cache device-grant auth (default: .pelican_auth_cache).",
    )
    parser.add_argument(
        "--oidc-url",
        default="https://token-issuer.icecube.aq",
        help="OIDC issuer base URL (default: https://token-issuer.icecube.aq).",
    )
    parser.add_argument(
        "--federation",
        default="pelican://osg-htc.org",
        help="Pelican federation name (default: pelican://osg-htc.org).",
    )
    parser.add_argument(
        "--storage-prefix",
        default="/icecube/wipac",
        help="Prefix to prepend to --target-path when writing (default: /icecube/wipac).",
    )
    parser.add_argument(
        "--with-write-scope",
        action="store_true",
        help=(
            "Request write/modify scope in addition to read scope, without performing the test write. "
            "Useful for workflows that need to write checkpoints but not run the test write."
        ),
    )
    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument(
        "--write",
        action="store_true",
        help="Perform the test write after fetching the token (default: token-only).",
    )
    write_group.add_argument(
        "--no-write",
        action="store_true",
        help="Deprecated alias for the default behavior (token-only).",
    )

    parser.add_argument(
        "--ls-source",
        action="store_true",
        help="List --source-path via PelicanFS and print the results (debug).",
    )
    parser.add_argument(
        "--ls-max-entries",
        type=int,
        default=50,
        help="Max entries to print for --ls-source (default: 50).",
    )

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data",
        help="Data to write when --write is set. If omitted, reads from stdin.",
    )
    data_group.add_argument(
        "--data-file",
        type=Path,
        help="File whose contents will be written.",
    )

    args = parser.parse_args(argv)

    target_path = args.target_path or args.source_path
    want_modify = bool(args.write or args.with_write_scope)

    # Default is token-only unless --write is explicitly requested.
    if not args.write:
        token = asyncio.run(
            get_access_token_async(
                oidc_url=args.oidc_url,
                source_path=args.source_path,
                target_path=target_path,
                auth_cache_file=args.auth_cache_file,
                want_modify=want_modify,
            )
        )
        print(token)
        return 0

    if args.data is not None:
        data = args.data
    elif args.data_file is not None:
        data = args.data_file.read_text(encoding="utf-8")
    else:
        # If stdin is a TTY, sys.stdin.read() blocks waiting for EOF and looks like a hang.
        # In that case, require the user to pass --no-write or provide data explicitly.
        try:
            is_tty = sys.stdin.isatty()
        except Exception:
            is_tty = False
        if is_tty:
            parser.error(
                "No data provided and stdin is interactive. Use token-only mode (default) or pass --data/--data-file, "
                "or pass --data/--data-file, or pipe stdin."
            )
        data = sys.stdin.read()

    token = asyncio.run(
        pelican(
            source_path=args.source_path,
            target_path=target_path,
            data=data,
            auth_cache_file=args.auth_cache_file,
            oidc_url=args.oidc_url,
            federation=args.federation,
            storage_prefix=args.storage_prefix,
            ls_source=args.ls_source,
            ls_max_entries=args.ls_max_entries,
        )
    )
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())