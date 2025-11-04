#!/usr/bin/env python3
"""
Test script to verify duplicate column handling in CarOBDMLDataLoader using actual CarOBD dataset.

This script tests the _handle_duplicate_columns method with real CarOBD data to ensure it correctly
handles duplicate columns that may arise when concatenating multiple CSV files.
"""

import sys
import os
import pandas as pd

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_anomaly_detector import CarOBDMLDataLoader


def test_actual_carobd_dataset():
    """Test with actual CarOBD dataset files."""
    print("=" * 70)
    print("Test: Actual CarOBD Dataset Integration Test")
    print("=" * 70)
    
    # Get the data path relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, "data", "carOBD", "obdiidata")
    
    if not os.path.exists(data_path):
        print(f"⚠️  Data path not found: {data_path}")
        print("   Skipping test - data directory does not exist")
        print("   This test requires the actual CarOBD dataset")
        return "SKIPPED"
    
    print(f"Loading CarOBD data from: {data_path}")
    
    try:
        # Initialize the loader with actual data path
        loader = CarOBDMLDataLoader(data_path)
        
        # Load all data (this will internally handle duplicate columns)
        idle_data, motion_data = loader.load_all_data()
        
        print(f"\nLoaded data:")
        print(f"  Idle data shape: {idle_data.shape}")
        print(f"  Motion data shape: {motion_data.shape}")
        print(f"  Idle columns ({len(idle_data.columns)}): {list(idle_data.columns[:5])}...")
        print(f"  Motion columns ({len(motion_data.columns)}): {list(motion_data.columns[:5])}...")
        
        # Verify no duplicate columns exist after loading
        idle_has_duplicates = idle_data.columns.duplicated().any()
        motion_has_duplicates = motion_data.columns.duplicated().any()
        
        print(f"\nDuplicate column check:")
        print(f"  Idle data has duplicates: {idle_has_duplicates}")
        print(f"  Motion data has duplicates: {motion_has_duplicates}")
        
        # Assertions
        assert not idle_has_duplicates, "Idle data should have no duplicate columns after loading"
        assert not motion_has_duplicates, "Motion data should have no duplicate columns after loading"
        
        # Verify expected columns are present
        expected_columns = [
            'COOLANT_TEMPERATURE ()',
            'ENGINE_RPM ()',
            'VEHICLE_SPEED ()',
            'THROTTLE ()',
            'ENGINE_LOAD ()'
        ]
        
        for col in expected_columns:
            assert col in idle_data.columns, f"Expected column '{col}' not found in idle data"
            assert col in motion_data.columns, f"Expected column '{col}' not found in motion data"
        
        print(f"\n✅ All expected columns present in both datasets")
        print(f"✅ No duplicate columns found")
        print(f"✅ Data shapes are valid: idle={idle_data.shape}, motion={motion_data.shape}")
        
        # Test that we can manually create duplicates and the handler fixes them
        print(f"\n--- Testing manual duplicate injection ---")
        
        # Create a copy of idle data and manually add duplicate columns
        test_df = idle_data.head(100).copy()
        
        # Add a duplicate column by copying an existing one
        test_df['COOLANT_TEMPERATURE ()_dup'] = test_df['COOLANT_TEMPERATURE ()'].copy()
        test_df.columns = list(test_df.columns[:-1]) + ['COOLANT_TEMPERATURE ()']
        
        print(f"Test DataFrame with duplicates:")
        print(f"  Shape: {test_df.shape}")
        print(f"  Columns: {list(test_df.columns)}")
        print(f"  Has duplicates: {test_df.columns.duplicated().any()}")
        
        # Apply duplicate handling
        processed = loader._handle_duplicate_columns(test_df)
        
        print(f"\nAfter duplicate handling:")
        print(f"  Shape: {processed.shape}")
        print(f"  Columns: {list(processed.columns)}")
        print(f"  Has duplicates: {processed.columns.duplicated().any()}")
        
        assert not processed.columns.duplicated().any(), "Manual duplicate injection should be handled"
        assert processed.shape[1] < test_df.shape[1], "Processed DataFrame should have fewer columns"
        
        print(f"✅ Manual duplicate injection test passed")
        
        # Test with 2 duplicates (should keep first)
        print(f"\n--- Testing 2 duplicates (should keep first) ---")
        test_df2 = idle_data.head(50).copy()
        # Find the index of COOLANT_TEMPERATURE column
        coolant_idx = test_df2.columns.get_loc('COOLANT_TEMPERATURE ()')
        # Store original values
        original_values = test_df2['COOLANT_TEMPERATURE ()'].copy()
        # Add duplicate at the end
        test_df2['COOLANT_TEMP_DUP'] = test_df2['COOLANT_TEMPERATURE ()'].copy() + 100  # Make it different
        test_df2.columns = list(test_df2.columns[:-1]) + ['COOLANT_TEMPERATURE ()']
        
        processed2 = loader._handle_duplicate_columns(test_df2)
        # Verify first occurrence is kept (original values, not the modified duplicate)
        assert processed2['COOLANT_TEMPERATURE ()'].equals(
            original_values
        ), "First duplicate should be kept when there are 2 duplicates"
        print("✅ 2 duplicates test passed (first kept)")
        
        # Test with 3 duplicates (should keep middle)
        print("\n--- Testing 3 duplicates (should keep middle) ---")
        test_df3 = idle_data.head(30).copy()
        original_values = test_df3['COOLANT_TEMPERATURE ()'].copy()
        # Add two duplicates with different values
        test_df3['COOLANT_TEMP_DUP1'] = test_df3['COOLANT_TEMPERATURE ()'].copy() + 200
        test_df3['COOLANT_TEMP_DUP2'] = test_df3['COOLANT_TEMPERATURE ()'].copy() + 300
        # Set column names to create 3 duplicates
        cols = list(test_df3.columns)
        cols[-2] = 'COOLANT_TEMPERATURE ()'
        cols[-1] = 'COOLANT_TEMPERATURE ()'
        test_df3.columns = cols
        
        processed3 = loader._handle_duplicate_columns(test_df3)
        assert not processed3.columns.duplicated().any(), "3 duplicates should be handled"
        assert 'COOLANT_TEMPERATURE ()' in processed3.columns, "COOLANT_TEMPERATURE should be present"
        # Middle duplicate should be kept (the one with +200)
        middle_values = original_values + 200
        assert processed3['COOLANT_TEMPERATURE ()'].equals(middle_values), "Middle duplicate should be kept when there are 3 duplicates"
        print("✅ 3 duplicates test passed (middle kept)")
        
        print("\n✅ All tests PASSED: Actual CarOBD dataset integration works correctly")
        print()
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED: Error loading CarOBD dataset")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Testing Duplicate Column Handling in CarOBDMLDataLoader")
    print("Using Actual CarOBD Dataset")
    print("=" * 70)
    print()
    
    tests = [
        test_actual_carobd_dataset
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            result = test()
            # Check if test was skipped (returns "SKIPPED")
            if result == "SKIPPED":
                skipped += 1
            else:
                passed += 1
        except AssertionError as e:
            print(f"❌ Test FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Test ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} {'❌' if failed > 0 else ''}")
    if skipped > 0:
        print(f"Skipped: {skipped} ⚠️")
    print("=" * 70)
    
    if failed == 0:
        if skipped > 0:
            print(f"\n🎉 All available tests passed! ({skipped} test(s) skipped - data not available)")
        else:
            print("\n🎉 All tests passed! Duplicate column handling works correctly with CarOBD dataset.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
