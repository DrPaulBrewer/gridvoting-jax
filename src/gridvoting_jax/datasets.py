
import os
import zipfile
import io
import shutil
from pathlib import Path
from typing import Optional

# Cache directory configuration
# Allow override via environment variable, default to /tmp/gridvoting_bjm_cache
BJM_CACHE_DIR = Path(os.environ.get('GV_BJM_CACHE_DIR', '/tmp/gridvoting_bjm_cache'))

def fetch_bjm_spatial_voting_2022_a100() -> Path:
    """
    Ensure BJM spatial voting data (2022 replication on A100) is downloaded and cached.
    
    Data Source:
    Brewer, P., Juybari, J. & Moberly, R. (2023). A comparison of zero- and minimal-intelligence 
    agendas in spatial voting games. BJM. doi:10.17605/BJM.IO/KMS9Z
    
    Returns:
        Path to the cache directory containing the downloaded CSV files.
    """
    # Check if cache exists and appears fully populated
    # We check for existence of all 8 expected configuration files to be safe
    expected_files = []
    for g in [20, 40, 60, 80]:
        for mode in ['MI', 'ZI']:
            expected_files.append(f'{g}_{mode}_stationary_distribution.csv')
            
    if BJM_CACHE_DIR.exists():
        existing_files = [f.name for f in BJM_CACHE_DIR.glob('*_*_stationary_distribution.csv')]
        # If we have at least 8 relevant files, assume cache is good/usable
        # (Exact matching every file might be too strict if user deleted one, but safer to re-download if missing)
        if len(existing_files) >= 8:
            return BJM_CACHE_DIR
    
    # Lazy import requests to avoid hard dependency on non-standard lib if not needed
    try:
        import requests
    except ImportError:
        # If requests is missing, we can't download. Return existing dir (best effort) or raise warning.
        # We'll just return the dir and let caller handle missing files.
        import warnings
        warnings.warn("gridvoting-jax: 'requests' module not installed. Cannot download BJM datasets.")
        return BJM_CACHE_DIR

    # Download and cache
    print(f"Downloading BJM data to cache: {BJM_CACHE_DIR}")
    BJM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    download_url_primary = "https://www.eaftc.com/data/pub/Brewer-Juybari-Moberly-JEIC-2023/bjm_spatial_triangle.zip"
    download_url_secondary = "https://osf.io/download/kms9z/"
    
    # Try primary URL first, then fall back to secondary
    download_successful = False
    for download_url in [download_url_primary, download_url_secondary]:
        try:
            response = requests.get(download_url, timeout=120) # Increased timeout for large files
            response.raise_for_status()
            
            # Extract zip file
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            
            # Find Run2 directory files in zip
            # (eaftc.com zip file has osf_cache subdir) 
            # (OSF zip file has Run2 subdir)
            data_files = [name for name in zip_file.namelist() 
                if 'Run2' in name or 'osf_cache' in name 
                and name.endswith('_stationary_distribution.csv')]
            
            # Extract stationary distribution files
            count = 0
            for filename in data_files:
                # Extract to cache with simplified name (flatten directory structure)
                basename = Path(filename).name
                with zip_file.open(filename) as source:
                    dest_path = BJM_CACHE_DIR / basename
                    with open(dest_path, 'wb') as dest:
                        dest.write(source.read())
                count += 1
            
            print(f"  Downloaded and extracted {count} stationary distribution files")
            download_successful = True
            break  # Success, exit the loop
            
        except Exception as e:
            # If this was the primary URL, try secondary
            if download_url == download_url_primary:
                import warnings
                warnings.warn(f"Failed to download from primary source: {e}. Trying secondary source...")
                continue
            # If this was the secondary URL (last attempt), warn and give up
            else:
                import warnings
                warnings.warn(f"Failed to download BJM data from all sources: {e}")
                # We don't raise here to allow library usage without internet, 
                # but downstream benchmarks will fail if they strictly need this data.
    
    return BJM_CACHE_DIR
