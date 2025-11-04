"""
Run KGroot with Bank dataset
This script processes the Bank data and runs KGroot training
"""

import os
import sys
import logging
from bank_data_processor import BankDataProcessor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def process_bank_data():
    """Process Bank telemetry data into KGroot format"""
    print("=" * 60)
    print("PROCESSING BANK DATA FOR KGROOT")
    print("=" * 60)
    
    bank_data_dir = "Bank"
    output_dir = "bank_from_bank"
    
    if not os.path.exists(bank_data_dir):
        print(f"Error: Bank data directory '{bank_data_dir}' not found!")
        return False
    
    processor = BankDataProcessor(bank_data_dir, output_dir)
    
    print("Processing Bank telemetry data...")
    try:
        processed_dates = processor.process_all_dates()
        
        print("Creating labeled data...")
        labeled_data_path = processor.create_labeled_data(processed_dates)
        
        print(f"✅ Processing complete! Output saved to: {output_dir}")
        print(f"✅ Processed {len(processed_dates)} dates")
        return True
        
    except Exception as e:
        print(f"❌ Error processing Bank data: {e}")
        return False

def run_kgroot():
    """Run KGroot training with processed Bank data"""
    print("\n" + "=" * 60)
    print("RUNNING KGROOT TRAINING")
    print("=" * 60)
    
    # Check if processed data exists
    if not os.path.exists("bank_from_bank/labeled_data.json"):
        print("❌ Error: Processed data not found. Run data processing first.")
        return False
    
    # Import and run KGroot
    try:
        # Fix import issue by adding current directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Import KGroot main class
        from graph_sim_dej_X import ModelInference
        
        print("Initializing KGroot model...")
        model = ModelInference()
        
        print("Starting training...")
        model.train_model()
        
        print("✅ KGroot training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error running KGroot: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function"""
    setup_logging()
    
    print("KGroot Implementation for Bank Dataset")
    print("=====================================")
    
    # Step 1: Process Bank data
    if not process_bank_data():
        print("❌ Data processing failed. Exiting.")
        return
    
    # Step 2: Run KGroot training
    if not run_kgroot():
        print("❌ KGroot training failed. Exiting.")
        return
    
    print("\n🎉 SUCCESS! KGroot implementation completed successfully!")

if __name__ == "__main__":
    main()
