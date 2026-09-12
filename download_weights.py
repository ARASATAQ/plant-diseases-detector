"""
Download pre-trained PlantVillage ResNet50 weights.
Run once before starting the server for full 38-class disease name classification.

Usage:
    python download_weights.py
"""
import os
import sys
import urllib.request

WEIGHTS_DIR  = 'weights'
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, 'plant_model.pth')


def _progress(block_num, block_size, total_size):
    if total_size > 0:
        pct = min(block_num * block_size / total_size * 100, 100)
        bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
        print(f'\r   [{bar}] {pct:.1f}%', end='', flush=True)


def try_huggingface() -> bool:
    """Attempt download from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        print('[1/2] Trying HuggingFace Hub…')
        path = hf_hub_download(
            repo_id='Diginsa/Plant-Disease-Detection-Project',
            filename='plant_disease_model_complete.pt',
            local_dir=WEIGHTS_DIR,
            local_dir_use_symlinks=False,
        )
        shutil.copy(path, WEIGHTS_FILE)
        print(f'      ✅ Saved to {WEIGHTS_FILE}')
        return True
    except Exception as exc:
        print(f'      ⚠️  HuggingFace failed: {exc}')
        return False


def try_direct() -> bool:
    """Fallback: try known direct download URLs."""
    urls = [
        ('GitHub releases',
         'https://github.com/imskr/Plant_Disease_Detection/releases/download/v1.0/plant_disease_model.pth'),
    ]
    for label, url in urls:
        try:
            print(f'[2/2] Trying {label}…\n   {url}')
            urllib.request.urlretrieve(url, WEIGHTS_FILE, reporthook=_progress)
            print(f'\n      ✅ Saved to {WEIGHTS_FILE}')
            return True
        except Exception as exc:
            print(f'\n      ⚠️  Failed: {exc}')
    return False


if __name__ == '__main__':
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if os.path.exists(WEIGHTS_FILE):
        size_mb = os.path.getsize(WEIGHTS_FILE) / 1_000_000
        print(f'✅ Weights already present ({size_mb:.1f} MB) — nothing to do.')
        print('   Run: python run.py')
        sys.exit(0)

    print('\n🌿 Plant Disease Detector — Weight Downloader\n')

    success = try_huggingface() or try_direct()

    if success:
        size_mb = os.path.getsize(WEIGHTS_FILE) / 1_000_000
        print(f'\n✅ Ready! ({size_mb:.1f} MB at {WEIGHTS_FILE})')
        print('   Now run: python run.py')
    else:
        print('\n❌ Automatic download failed.')
        print('   The app still works via colour-based analysis (no weights needed).')
        print('   See README.md §Manual Download for alternative instructions.')
        sys.exit(1)
