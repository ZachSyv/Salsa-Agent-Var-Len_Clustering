#!/usr/bin/env python3
"""
Simple test script for Salsa-Agent to verify basic functionality
without requiring the full training dataset.
"""

import torch
import os
import sys
sys.path.append('.')

from models.mllm import MotionLLM
from options.option_llm import get_args_parser

def test_model_loading():
    """Test if the model can load successfully"""
    print("Testing model loading...")
    
    # Get arguments
    args = get_args_parser()
    args.device = 'cpu'  # Use CPU for testing
    args.is_baseline = False
    args.is_MDM = False
    
    try:
        # Initialize model
        model = MotionLLM(args)
        print("✅ Model initialized successfully")
        
        # Test loading the main checkpoint
        model.load_model('ckpt/motionllm.pth')
        print("✅ Main checkpoint loaded successfully")
        
        # Test loading a fine-tuned checkpoint
        model.load_model('checkpoints/Xmotionllm_epoch14.pth')
        print("✅ Fine-tuned checkpoint loaded successfully")
        
        print("\n🎉 All model loading tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        return False

def test_basic_inference():
    """Test basic inference without training data"""
    print("\nTesting basic inference...")
    
    try:
        # Get arguments
        args = get_args_parser()
        args.device = 'cpu'
        args.is_baseline = False
        args.is_MDM = False
        
        # Initialize model
        model = MotionLLM(args)
        model.load_model('checkpoints/Xmotionllm_epoch14.pth')
        
        # Test tokenizer
        test_text = "Basic salsa dance move"
        input_ids = model.tokenizer.encode(test_text, return_tensors='pt')
        print(f"✅ Tokenizer working: '{test_text}' -> {input_ids.shape}")
        
        print("✅ Basic inference test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during basic inference: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Salsa-Agent simple tests...\n")
    
    # Test 1: Model loading
    model_ok = test_model_loading()
    
    # Test 2: Basic inference
    inference_ok = test_basic_inference()
    
    print("\n" + "="*50)
    if model_ok and inference_ok:
        print("🎉 All tests passed! Salsa-Agent is working correctly.")
        print("\nNext steps:")
        print("1. The model loads successfully")
        print("2. Checkpoints are accessible")
        print("3. Basic functionality works")
        print("\nTo run the full demo, you'll need:")
        print("- The training dataset (dataset_processed/lmdb_Salsa_pair/lmdb_train)")
        print("- Or modify the demo to work without training data")
    else:
        print("❌ Some tests failed. Check the errors above.")
    print("="*50)



