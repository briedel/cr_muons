from rest_tools.client.openid_client import OpenIDRestClient
from rest_tools.client.device_client import CommonDeviceGrant
from rest_tools.utils.auth import OpenIDAuth
from rest_tools.client.openid_client import RegisterOpenIDClient
from pelicanfs import PelicanFileSystem
from uuid import uuid4
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import jwt


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
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _get_access_token_async(
                oidc_url=oidc_url,
                source_path=source_path,
                target_path=target_path,
                auth_cache_file=auth_cache_file,
            )
        )
    raise RuntimeError(
        "_get_access_token() cannot be called from an event loop; use await _get_access_token_async(...)"
    )


async def _get_access_token_async(
    *,
    oidc_url: str,
    source_path: str,
    target_path: str,
    auth_cache_file: str,
) -> str:
    source_path = _normalize_scope_path(source_path)
    target_path = _normalize_scope_path(target_path)

    scopes = [
        f"storage.modify:{target_path}",
        f"storage.read:{source_path}",
    ]
    requested_scopes = set(scopes)

    cache_path = Path(auth_cache_file)

    def load_cache() -> dict[str, Any]:
        if not cache_path.exists():
            return {}
        raw = cache_path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        # Backward-compat: older cache files may contain only a refresh token string.
        if not raw.startswith("{"):
            return {"refresh_token": raw}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {"refresh_token": raw}

    def save_cache(data: dict[str, Any]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cache = load_cache()

    client_id = cache.get("client_id")
    client_secret = cache.get("client_secret")
    refresh_token = cache.get("refresh_token")

    cached_oidc_url = cache.get("oidc_url")
    if cached_oidc_url and cached_oidc_url != oidc_url:
        # Avoid trying to refresh against the wrong issuer.
        refresh_token = None

    if not client_id or not client_secret:
        reg_client = RegisterOpenIDClient(f"{oidc_url}", str(uuid4().hex))
        client_id, client_secret = await reg_client.register_client()
        cache["oidc_url"] = oidc_url
        cache["client_id"] = client_id
        cache["client_secret"] = client_secret
        save_cache(cache)

    def update_func(access_token: Any, new_refresh: Any) -> None:
        if new_refresh:
            cache["refresh_token"] = new_refresh
            save_cache(cache)

    def _get_refresh_token_scopes(token: str) -> set[str] | None:
        cached_scopes = cache.get("scopes")
        if isinstance(cached_scopes, list) and all(isinstance(s, str) for s in cached_scopes):
            return set(cached_scopes)
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            scope_str = decoded.get("scope", "")
            if not isinstance(scope_str, str):
                return None
            scopes_from_token = {s for s in scope_str.split() if s}
            return scopes_from_token or None
        except jwt.exceptions.DecodeError:
            return None

    def _storage_scopes(scope_set: set[str]) -> set[str]:
        return {s for s in scope_set if s.startswith("storage.")}

    if refresh_token:
        token_scopes = _get_refresh_token_scopes(refresh_token)
        if token_scopes is not None:
            # Enforce exact match for storage.* scopes. (Other OIDC scopes like offline_access may exist.)
            if _storage_scopes(token_scopes) != requested_scopes:
                refresh_token = None

    # First, try to use the cached refresh token if present.
    if refresh_token:
        try:
            client = OpenIDRestClient(
                address="",
                token_url=oidc_url,
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                update_func=update_func,
            )
            return client._openid_token()
        except Exception:
            # fall back to device grant below
            pass

    # Otherwise do device flow to obtain a refresh token, then exchange for access token.
    auth = OpenIDAuth(oidc_url)
    if not auth.provider_info:
        raise RuntimeError("Token service does not support .well-known discovery")
    if "device_authorization_endpoint" not in auth.provider_info:
        raise RuntimeError("Device grant not supported by server")
    device_endpoint: str = auth.provider_info["device_authorization_endpoint"]  # type: ignore[assignment]

    device = CommonDeviceGrant()
    refresh_token = device.perform_device_grant(
        logger=logging.getLogger("SavedDeviceGrantAuth"),
        device_url=device_endpoint,
        token_url=auth.token_url,
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
    )
    cache["refresh_token"] = refresh_token
    cache["scopes"] = sorted(requested_scopes)
    save_cache(cache)

    client = OpenIDRestClient(
        address="",
        token_url=oidc_url,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        update_func=update_func,
    )
    return client._openid_token()


async def pelican(
    source_path: str,
    target_path: str,
    data: str,
    auth_cache_file: str = ".pelican_auth_cache",
    oidc_url: str = "https://token-issuer.icecube.aq",
    federation: str = "osdf://",
    storage_prefix: str = "/icecube/wipac",
) -> str:
    source_path = _normalize_scope_path(source_path)
    target_path = _normalize_scope_path(target_path)
    storage_prefix = storage_prefix.rstrip("/")
    full_path = f"{storage_prefix}{target_path}"

    access_token = await _get_access_token_async(
        oidc_url=oidc_url,
        source_path=source_path,
        target_path=target_path,
        auth_cache_file=auth_cache_file,
    )

    print(federation, full_path)

    pelfs = PelicanFileSystem(
        federation,
        headers={"Authorization": f"Bearer {access_token}"},
        direct_reads=True,
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
        default="osdf://",
        help="Pelican federation name (default: osdf://).",
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
        token = asyncio.run(
            _get_access_token_async(
                oidc_url=args.oidc_url,
                source_path=args.source_path,
                target_path=args.target_path,
                auth_cache_file=args.auth_cache_file,
            )
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