#!/usr/bin/env python3
"""
Test script to verify all imports work correctly after reorganization.
Run this to ensure the code can execute without import errors.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_goat_package():
    """Test main goat package imports"""
    print("Testing goat package imports...")
    try:
        import goat
        print("✓ goat package")
    except Exception as e:
        print(f"✗ goat package: {e}")
        return False
    return True

def test_model_imports():
    """Test model imports"""
    print("\nTesting model imports...")
    success = True
    
    try:
        from goat.models.networks.Methods.GOAT_T import GOAT_T
        print("✓ GOAT_T")
    except Exception as e:
        print(f"✗ GOAT_T: {e}")
        success = False
    
    try:
        from goat.models.networks.Methods.GOAT_L import GOAT_L
        print("✓ GOAT_L")
    except Exception as e:
        print(f"✗ GOAT_L: {e}")
        success = False
    
    return success

def test_loss_imports():
    """Test loss function imports"""
    print("\nTesting loss imports...")
    success = True
    
    try:
        from goat.losses.modules.disparity_sequence_loss import sequence_lossV2
        print("✓ disparity_sequence_loss")
    except Exception as e:
        print(f"✗ disparity_sequence_loss: {e}")
        success = False
    
    try:
        from goat.losses.modules.entropy_loss import compute_entropy_loss
        print("✓ entropy_loss")
    except Exception as e:
        print(f"✗ entropy_loss: {e}")
        success = False
    
    try:
        from goat.losses.modules.multi_scale_loss import MultiScaleLoss
        print("✓ multi_scale_loss")
    except Exception as e:
        print(f"✗ multi_scale_loss: {e}")
        success = False
    
    return success

def test_utils_imports():
    """Test utility imports"""
    print("\nTesting utils imports...")
    success = True
    
    try:
        from goat.utils.core.common import load_loss_scheme, logger
        print("✓ common utils")
    except Exception as e:
        print(f"✗ common utils: {e}")
        success = False
    
    try:
        from goat.utils.core.metric import compute_iou, Disparity_EPE_Loss
        print("✓ metrics")
    except Exception as e:
        print(f"✗ metrics: {e}")
        success = False
    
    try:
        from goat.utils.core.AverageMeter import AverageMeter
        print("✓ AverageMeter")
    except Exception as e:
        print(f"✗ AverageMeter: {e}")
        success = False
    
    return success

def test_trainer_import():
    """Test trainer import"""
    print("\nTesting trainer import...")
    try:
        from goat.trainer import DisparityTrainer
        print("✓ DisparityTrainer")
        return True
    except Exception as e:
        print(f"✗ DisparityTrainer: {e}")
        return False

def test_dataloader_imports():
    """Test dataloader imports"""
    print("\nTesting dataloader imports...")
    success = True
    
    try:
        from data.dataloaders.kitti.SceneflowLoaderOcc import StereoDatasetOcc
        print("✓ SceneflowLoaderOcc")
    except Exception as e:
        print(f"✗ SceneflowLoaderOcc: {e}")
        success = False
    
    try:
        from data.dataloaders.kitti.kitti_loader import KITTILoader
        print("✓ KITTILoader")
    except Exception as e:
        print(f"✗ KITTILoader: {e}")
        success = False
    
    try:
        from data.dataloaders.fat.FAT_dataset import FATDataset
        print("✓ FATDataset")
    except Exception as e:
        print(f"✗ FATDataset: {e}")
        success = False
    
    return success

def main():
    """Run all import tests"""
    print("="*60)
    print("GOAT Import Verification Test")
    print("="*60)
    
    all_success = True
    
    # Run all tests
    all_success &= test_goat_package()
    all_success &= test_model_imports()
    all_success &= test_loss_imports()
    all_success &= test_utils_imports()
    all_success &= test_trainer_import()
    all_success &= test_dataloader_imports()
    
    print("\n" + "="*60)
    if all_success:
        print("✓ ALL IMPORTS SUCCESSFUL!")
        print("The code is ready to run on GPUs.")
    else:
        print("✗ SOME IMPORTS FAILED!")
        print("Please check the error messages above.")
    print("="*60)
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())

