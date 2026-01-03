from rest_tools.client import RegisterOpenIDClient, SavedDeviceGrantAuth
from pelicanfs import PelicanFileSystem
from uuid import uuid4

async def pelican(source_path, target_path, data, 
                  auth_cache_file=".pelican_auth_cache"):
    oidc_url = "https://token-issuer.icecube.aq"
    full_path = f"/icecube/wipac{target_path}"
    
    # 1. Register a dynamic client
    # This utility handles the OIDC Dynamic Client Registration protocol
    reg_client = RegisterOpenIDClient(oidc_url+"/client")
    client_info = reg_client.register(
        client_name=uuid4().hex,
        redirect_uris=[] # Device grant flows typically don't require redirects
    )
    
    # Extract credentials from registration response
    client_id = client_info["client_id"]
    client_secret = client_info["client_secret"]
    
    # 2. Obtain an access token via SavedDeviceGrantAuth
    # It uses the newly registered client_id/secret and caches the token locally
    auth_client = SavedDeviceGrantAuth(
        address="",
        token_url=oidc_url,
        client_id=client_id,
        client_secret=client_secret,
        filename=auth_cache_file,
        scopes=[f"storage.modify:{target_path}", f"storage.read:{source_path}"],
    )
    
    # Get the token (triggers device flow if no valid cached token exists)
    access_token = auth_client.get_access_token()
    
    # 3. Write to PelicanFS using the token in the header
    # PelicanFS accepts standard 'Authorization' headers for protected writes
    pelfs = PelicanFileSystem(
        "osdf", 
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    with pelfs.open(full_path, "w") as f:
        f.write(data)
    print(f"Successfully wrote data to Pelican path: {full_path}")

    return access_token