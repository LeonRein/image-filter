#!/usr/bin/env python3
"""
Test script for the database functionality of the image filter tool.
This script tests the database operations without processing actual images.
"""

import os
import sys
import tempfile
import sqlite3
import json
from datetime import datetime

# Add the project directory to the path to import from main.py
sys.path.insert(0, '/home/hans/projects/image-filter')

# Mock the configuration for testing
TEST_CONFIG = {
    'source_folder': "~/test/",
    'target_folder': "~/test/discarded/",
    'model_path': "yolov8s.pt",
    'device': 'cpu',
    'target_classes': ['car', 'person'],
    'large_area_threshold': 200000,
    'small_area_threshold': 50000,
    'min_confidence_threshold': 0.25,
    'max_confidence_threshold': 0.7,
    'min_resolution': [1920, 1080],
    'aspect_ratio_tolerance': 0.1,
    'supported_extensions': (".jpg", ".jpeg", ".png")
}

def test_database_functions():
    """Test the database functionality"""
    import main
    
    # Override the CONFIG for testing
    main.CONFIG = TEST_CONFIG
    main.source_folder = "/tmp/test_image_filter"
    
    # Create a temporary test directory
    os.makedirs(main.source_folder, exist_ok=True)
    
    print("🧪 Testing database functionality...")
    
    try:
        # Test 1: Database initialization
        print("1️⃣ Testing database initialization...")
        db_path = main.init_database()
        assert os.path.exists(db_path), "Database file was not created"
        print("   ✅ Database created successfully")
        
        # Test 2: Configuration hash
        print("2️⃣ Testing configuration hash...")
        hash1 = main.get_config_hash()
        hash2 = main.get_config_hash()
        assert hash1 == hash2, "Configuration hash is not consistent"
        print(f"   ✅ Configuration hash: {hash1[:8]}...")
        
        # Test 3: Save and retrieve file result
        print("3️⃣ Testing file result saving and retrieval...")
        test_file = "test_image.jpg"
        test_size = 12345
        test_mtime = 1234567890.123
        test_decision = "kept"
        test_reasons = ["no_objects_found"]
        
        # Save result
        main.save_file_result(db_path, test_file, test_size, test_mtime, test_decision, test_reasons)
        
        # Retrieve result
        result = main.is_file_processed(db_path, test_file, test_size, test_mtime)
        assert result is not None, "Could not retrieve saved result"
        
        decision, reasons_json = result
        reasons = json.loads(reasons_json) if reasons_json else []
        
        assert decision == test_decision, f"Decision mismatch: {decision} != {test_decision}"
        assert reasons == test_reasons, f"Reasons mismatch: {reasons} != {test_reasons}"
        print("   ✅ File result save/retrieve works correctly")
        
        # Test 4: Configuration change detection
        print("4️⃣ Testing configuration change detection...")
        
        # Change configuration
        original_threshold = main.CONFIG['large_area_threshold']
        main.CONFIG['large_area_threshold'] = 999999
        
        # Initialize database again (should reset)
        old_hash = hash1
        new_hash = main.get_config_hash()
        assert old_hash != new_hash, "Configuration hash should change when config changes"
        print(f"   ✅ Configuration change detected: {old_hash[:8]}... -> {new_hash[:8]}...")
        
        # Test 5: Cache statistics
        print("5️⃣ Testing cache statistics...")
        main.show_cache_stats(db_path)
        print("   ✅ Cache statistics displayed successfully")
        
        print("\n🎉 All database tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(main.source_folder):
                os.rmdir(main.source_folder)
        except:
            pass
    
    return True

if __name__ == "__main__":
    success = test_database_functions()
    sys.exit(0 if success else 1)
