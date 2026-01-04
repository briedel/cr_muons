from rest_tools.client import SavedDeviceGrantAuth
from rest_tools.oidc import RegisterOpenIDClient
from pelicanfs import PelicanFileSystem
from uuid import uuid4
import argparse
import asyncio
import sys
from pathlib import Path


def _normalize_scope_path(p: str) -> str:
    if not p:
        return "/"
    return p if p.startswith("/") else f"/{p}"


def _get_access_token(
    *,
    oidc_url: str,
    source_path: str,
    target_path: str,
    auth_cache_file: str,
) -> str:
    source_path = _normalize_scope_path(source_path)
    target_path = _normalize_scope_path(target_path)

    reg_client = RegisterOpenIDClient(f"{oidc_url}/client")
    client_info = reg_client.register(
        client_name=uuid4().hex,
        redirect_uris=[],
    )

    client_id = client_info["client_id"]
    client_secret = client_info["client_secret"]

    auth_client = SavedDeviceGrantAuth(
        address="",
        token_url=oidc_url,
        client_id=client_id,
        client_secret=client_secret,
        filename=auth_cache_file,
        scopes=[
            f"storage.modify:{target_path}",
            f"storage.read:{source_path}",
        ],
    )

    return auth_client.get_access_token()

async def pelican(
    source_path: str,
    target_path: str,
    data: str,
    auth_cache_file: str = ".pelican_auth_cache",
    oidc_url: str = "https://token-issuer.icecube.aq",
    federation: str = "osdf",
    storage_prefix: str = "/icecube/wipac",
) -> str:
    source_path = _normalize_scope_path(source_path)
    target_path = _normalize_scope_path(target_path)
    storage_prefix = storage_prefix.rstrip("/")
    full_path = f"{storage_prefix}{target_path}"

    access_token = _get_access_token(
        oidc_url=oidc_url,
        source_path=source_path,
        target_path=target_path,
        auth_cache_file=auth_cache_file,
    )

    pelfs = PelicanFileSystem(
        federation,
        headers={"Authorization": f"Bearer {access_token}"},
    )

    with pelfs.open(full_path, "w") as f:
        f.write(data)
    print(f"Successfully wrote data to Pelican path: {full_path}")

    return access_token


def main(argv: list[str] | None = None) -> int:
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
        required=True,
        help="Path used to request modify scope and (by default) where data will be written.",
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
        default="osdf",
        help="Pelican federation name (default: osdf).",
    )
    parser.add_argument(
        "--storage-prefix",
        default="/icecube/wipac",
        help="Prefix to prepend to --target-path when writing (default: /icecube/wipac).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only fetch and print the token; do not write any data.",
    )

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data",
        help="Data to write. If omitted and not --no-write, reads from stdin.",
    )
    data_group.add_argument(
        "--data-file",
        type=Path,
        help="File whose contents will be written.",
    )

    args = parser.parse_args(argv)

    if args.no_write:
        token = _get_access_token(
            oidc_url=args.oidc_url,
            source_path=args.source_path,
            target_path=args.target_path,
            auth_cache_file=args.auth_cache_file,
        )
        print(token)
        return 0

    if args.data is not None:
        data = args.data
    elif args.data_file is not None:
        data = args.data_file.read_text(encoding="utf-8")
    else:
        data = sys.stdin.read()

    token = asyncio.run(
        pelican(
            source_path=args.source_path,
            target_path=args.target_path,
            data=data,
            auth_cache_file=args.auth_cache_file,
            oidc_url=args.oidc_url,
            federation=args.federation,
            storage_prefix=args.storage_prefix,
        )
    )
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())