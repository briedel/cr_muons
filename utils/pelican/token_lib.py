import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4



def normalize_scope_path(path: str) -> str:
    if not path:
        return "/"
    return path if path.startswith("/") else f"/{path}"


def get_access_token(
    *,
    oidc_url: str,
    source_path: str,
    target_path: str,
    auth_cache_file: str,
    want_modify: bool = True,
) -> str:
    """Fetch an access token using cached refresh token or interactive device flow.

    This is a synchronous wrapper around `get_access_token_async`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            get_access_token_async(
                oidc_url=oidc_url,
                source_path=source_path,
                target_path=target_path,
                auth_cache_file=auth_cache_file,
                want_modify=want_modify,
            )
        )
    raise RuntimeError(
        "get_access_token() cannot be called from an event loop; use await get_access_token_async(...)"
    )


async def get_access_token_async(
    *,
    oidc_url: str,
    source_path: str,
    target_path: str,
    auth_cache_file: str,
    want_modify: bool = True,
) -> str:
    logger = logging.getLogger("get_pelican_token")

    try:
        from rest_tools.client.device_client import CommonDeviceGrant
        from rest_tools.client.openid_client import OpenIDRestClient, RegisterOpenIDClient
        from rest_tools.utils.auth import OpenIDAuth
    except ImportError as e:
        raise ImportError(
            "Missing dependency for Pelican device-flow token acquisition. Install `wipac-rest-tools`."
        ) from e

    source_path = normalize_scope_path(source_path)
    target_path = normalize_scope_path(target_path)

    scopes = [f"storage.read:{source_path}"]
    if want_modify:
        scopes.insert(0, f"storage.modify:{target_path}")
    requested_scopes = set(scopes)

    logger.info(
        "Requesting Pelican token (want_modify=%s) with scopes: %s",
        want_modify,
        " ".join(scopes),
    )

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

    logger.info("Auth cache file: %s", str(cache_path))

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
        # Optional dependency: if PyJWT isn't installed, we cannot decode scope claims.
        try:
            import jwt  # type: ignore
        except Exception:
            return None
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            scope_str = decoded.get("scope", "")
            if not isinstance(scope_str, str):
                return None
            scopes_from_token = {s for s in scope_str.split() if s}
            return scopes_from_token or None
        except Exception:
            return None

    def _storage_scopes(scope_set: set[str]) -> set[str]:
        return {s for s in scope_set if s.startswith("storage.")}

    if refresh_token:
        token_scopes = _get_refresh_token_scopes(refresh_token)
        if token_scopes is not None:
            # Enforce exact match for storage.* scopes. (Other OIDC scopes like offline_access may exist.)
            if _storage_scopes(token_scopes) != requested_scopes:
                refresh_token = None

    if refresh_token:
        logger.info("Attempting to use cached refresh token")
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
            pass

    logger.info("Starting device flow")
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
