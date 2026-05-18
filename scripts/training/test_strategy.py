# scripts/training/test_strategy.py

import sys
import torch
from pathlib import Path

# ================== FIX PYTHON PATH ==================
# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up from scripts/training/
sys.path.append(str(PROJECT_ROOT))

print(f"Project root added to path: {PROJECT_ROOT}\n")

# Now import
from scripts.utils.dataset import SRDataset
from scripts.models.swinir import SwinIRModule   # Make sure this file exists


def test_strategies():
    print("🔍 Starting Model Strategy Tests...\n")
    
    PATCHES_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
    STATS_PATH = PATCHES_DIR / "stats.json"

    # Load one sample
    ds = SRDataset(
        split="train",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=True,
        aux_dir=str(PATCHES_DIR / "train" / "AUX"),
        use_water_mask=True,
        repeat_channels=False,
        transform=None,
    )

    sample = ds[0]
    print(f"LR shape : {sample['lr'].shape}")
    print(f"HR shape : {sample['hr'].shape}")
    print(f"AUX shape: {sample.get('aux', None).shape if 'aux' in sample else None}")
    
    aux_chans = sample['aux'].shape[0]
    print(f"→ Detected {aux_chans} auxiliary channels\n")

    # Test strategies
    strategies = ["projection", "direct", "fusion"]
    pretrained_path = str(Path("/share/home/e2406751/Superresolution-TIR/data/pretrained/SwinIR_classical_x4.pth"))

    for strategy in strategies:
        print(f"Testing strategy: **{strategy.upper()}**")
        
        try:
            model = SwinIRModule(
                pretrained_path=pretrained_path,
                learning_rate=1e-4,
                adaptation_strategy=strategy,
                aux_chans=aux_chans,
            )
            
            # Create batch
            batch = {
                "lr": sample["lr"].unsqueeze(0),
                "aux": sample["aux"].unsqueeze(0),
            }

            model.eval()
            with torch.no_grad():
                output = model(batch)

            print(f"   Output shape : {output.shape}")
            print(f"   ✅ {strategy} passed\n")
            
        except Exception as e:
            print(f"   ❌ {strategy} failed → {e}\n")
            import traceback
            traceback.print_exc()

    print("🎉 All tests finished!")


if __name__ == "__main__":
    test_strategies()