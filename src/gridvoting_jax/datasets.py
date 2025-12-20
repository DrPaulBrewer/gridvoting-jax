
import os
import zipfile
import io
import shutil
from pathlib import Path
from typing import Optional

# Cache directory configuration
# Allow override via environment variable, default to /tmp/gridvoting_osf_cache
OSF_CACHE_DIR = Path(os.environ.get('GV_OSF_CACHE_DIR', '/tmp/gridvoting_osf_cache'))
OSF_FILE_ID = 'kms9z'  # Zip file with A100 GPU replication data

def fetch_osf_spatial_voting_2022_a100() -> Path:
    """
    Ensure OSF spatial voting data (2022 replication on A100) is downloaded and cached.
    
    Data Source:
    Brewer, P., Juybari, J. & Moberly, R. (2023). A comparison of zero- and minimal-intelligence 
    agendas in spatial voting games. OSF. doi:10.17605/OSF.IO/KMS9Z
    
    Returns:
        Path to the cache directory containing the downloaded CSV files.
    """
    # Check if cache exists and appears fully populated
    # We check for existence of all 8 expected configuration files to be safe
    expected_files = []
    for g in [20, 40, 60, 80]:
        for mode in ['MI', 'ZI']:
            expected_files.append(f'{g}_{mode}_stationary_distribution.csv')
            
    if OSF_CACHE_DIR.exists():
        existing_files = [f.name for f in OSF_CACHE_DIR.glob('*_*_stationary_distribution.csv')]
        # If we have at least 8 relevant files, assume cache is good/usable
        # (Exact matching every file might be too strict if user deleted one, but safer to re-download if missing)
        if len(existing_files) >= 8:
            return OSF_CACHE_DIR
    
    # Lazy import requests to avoid hard dependency on non-standard lib if not needed
    try:
        import requests
    except ImportError:
        # If requests is missing, we can't download. Return existing dir (best effort) or raise warning.
        # We'll just return the dir and let caller handle missing files.
        import warnings
        warnings.warn("gridvoting-jax: 'requests' module not installed. Cannot download OSF datasets.")
        return OSF_CACHE_DIR

    # Download and cache
    print(f"Downloading OSF data to cache: {OSF_CACHE_DIR}")
    OSF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    download_url = f"https://osf.io/download/{OSF_FILE_ID}/"
    
    try:
        response = requests.get(download_url, timeout=120) # Increased timeout for large files
        response.raise_for_status()
        
        # Extract zip file
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        
        # Find Run2 directory files in zip (original paper structure had Run2 subdir)
        run2_files = [name for name in zip_file.namelist() 
                     if 'Run2' in name and name.endswith('_stationary_distribution.csv')]
        
        # Extract stationary distribution files
        count = 0
        for filename in run2_files:
            # Extract to cache with simplified name (flatten directory structure)
            basename = Path(filename).name
            with zip_file.open(filename) as source:
                dest_path = OSF_CACHE_DIR / basename
                with open(dest_path, 'wb') as dest:
                    dest.write(source.read())
            count += 1
        
        print(f"  Downloaded and extracted {count} stationary distribution files")
        
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to download OSF data: {e}")
        # We don't raise here to allow library usage without internet, 
        # but downstream benchmarks will fail if they strictly need this data.
    
    return OSF_CACHE_DIR
