"""
Bank Data Processor for KGroot
=================================

This module converts Bank telemetry data (logs, metrics, traces) into graph format 
required by the KGroot algorithm. The processor:

1. Loads telemetry data from CSV files (logs, metrics, traces)
2. Extracts components/services from the data
3. Creates node features based on telemetry statistics
4. Builds multiple adjacency matrices representing different relationships
5. Saves processed graphs as pickle files for KGroot training

Key Concepts:
- Components: Services, containers, or applications identified by cmdb_id or tc
- Features: 100-dimensional vectors representing component characteristics
- Adjacency Matrices: 3 different relationship types between components
- Graph Similarity: The goal is to learn if two system states are similar

Data Flow:
Raw Telemetry → Component Extraction → Feature Engineering → Graph Construction → Pickle Files
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import networkx as nx
from collections import defaultdict

class BankDataProcessor:
    def __init__(self, bank_data_dir: str, output_dir: str):
        """
        Initialize the Bank Data Processor
        
        Args:
            bank_data_dir: Directory containing Bank telemetry data
            output_dir: Directory to save processed graph data
        """
        self.bank_data_dir = bank_data_dir  # "Bank" directory with telemetry data
        self.output_dir = output_dir        # "bank_from_bank" output directory
        self.telemetry_dir = os.path.join(bank_data_dir, "telemetry")
        self.query_file = os.path.join(bank_data_dir, "query.csv")    # Query tasks
        self.record_file = os.path.join(bank_data_dir, "record.csv")  # Fault records
        
        # Create output directories for processed data
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pickle_data"), exist_ok=True)
        
        # Load ground truth data for fault detection
        self.fault_records = self._load_fault_records()  # Known fault timestamps
        self.queries = self._load_queries()              # Query tasks for evaluation
        
    def _load_fault_records(self) -> pd.DataFrame:
        """Load fault records with timestamps"""
        df = pd.read_csv(self.record_file)
        df['timestamp'] = pd.to_datetime(df['datetime'])
        return df
    
    def _load_queries(self) -> pd.DataFrame:
        """Load query tasks"""
        df = pd.read_csv(self.query_file)
        return df
    
    def _load_telemetry_data(self, date: str, sample_size: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Load telemetry data for a specific date (with sampling for faster processing)
        
        This function loads three types of telemetry data:
        1. Logs: Application/service logs with error messages and events
        2. Metrics: Performance metrics (response time, success rate, etc.)
        3. Traces: Distributed tracing data showing service call relationships
        
        Args:
            date: Date string in format "YYYY_MM_DD"
            sample_size: Number of rows to sample from each file (for performance)
            
        Returns:
            Dictionary containing 'logs', 'metrics_app', 'metrics_container', 'traces'
        """
        date_dir = os.path.join(self.telemetry_dir, date)
        data = {}
        
        # Load application logs - contain error messages and service events
        log_file = os.path.join(date_dir, "log", "log_service.csv")
        if os.path.exists(log_file):
            print(f"  Loading logs from {log_file}...")
            # Sample data for faster processing (telemetry data can be very large)
            data['logs'] = pd.read_csv(log_file, nrows=sample_size)
            print(f"  Loaded {len(data['logs'])} log entries")
        
        # Load application metrics - performance indicators like response time, success rate
        metric_app_file = os.path.join(date_dir, "metric", "metric_app.csv")
        if os.path.exists(metric_app_file):
            print(f"  Loading app metrics from {metric_app_file}...")
            data['metrics_app'] = pd.read_csv(metric_app_file, nrows=sample_size)
            print(f"  Loaded {len(data['metrics_app'])} app metric entries")
            
        # Load container metrics - system-level metrics like CPU, memory usage
        metric_container_file = os.path.join(date_dir, "metric", "metric_container.csv")
        if os.path.exists(metric_container_file):
            print(f"  Loading container metrics from {metric_container_file}...")
            data['metrics_container'] = pd.read_csv(metric_container_file, nrows=sample_size)
            print(f"  Loaded {len(data['metrics_container'])} container metric entries")
        
        # Load distributed traces - shows how requests flow between services
        trace_file = os.path.join(date_dir, "trace", "trace_span.csv")
        if os.path.exists(trace_file):
            print(f"  Loading traces from {trace_file}...")
            data['traces'] = pd.read_csv(trace_file, nrows=sample_size)
            print(f"  Loaded {len(data['traces'])} trace entries")
        
        return data
    
    def _create_component_graph(self, telemetry_data: Dict[str, pd.DataFrame], 
                               fault_time: datetime = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Create graph representation from telemetry data
        
        This is the CORE FUNCTION that converts raw telemetry data into graph format:
        
        1. COMPONENT EXTRACTION: Identify all services/containers from telemetry data
        2. FEATURE ENGINEERING: Create 100-dimensional feature vectors for each component
        3. RELATIONSHIP BUILDING: Create 3 different adjacency matrices representing:
           - Service call relationships (from traces)
           - Temporal correlations (from logs)
           - Metric correlations (from performance data)
        
        Args:
            telemetry_data: Dictionary containing logs, metrics, traces DataFrames
            fault_time: Optional timestamp when fault occurred
            
        Returns:
            Tuple of (node_features, adjacency_matrices)
            - node_features: (n_components, 100) - feature vectors for each component
            - adjacency_matrices: List of 3 adjacency matrices (n_components, n_components)
        """
        # STEP 1: Extract all unique components/services from telemetry data
        components = set()
        
        # Get components from different data sources using actual column names
        if 'logs' in telemetry_data:
            components.update(telemetry_data['logs']['cmdb_id'].unique())  # Service IDs from logs
        if 'metrics_app' in telemetry_data:
            components.update(telemetry_data['metrics_app']['tc'].unique())  # tc = service name
        if 'metrics_container' in telemetry_data:
            components.update(telemetry_data['metrics_container']['cmdb_id'].unique())  # Container IDs
        if 'traces' in telemetry_data:
            components.update(telemetry_data['traces']['cmdb_id'].unique())  # Service IDs from traces
        
        components = list(components)  # Convert to list for indexing
        n_components = len(components)
        
        if n_components == 0:
            # Create dummy graph if no components found (fallback)
            return self._create_dummy_graph()
        
        # STEP 2: Create node features (100-dimensional vectors as per KGroot config)
        input_dim = 100  # Must match config_graph_sim.ini input_dim
        features = np.zeros((n_components, input_dim), dtype=np.float32)
        
        # STEP 3: Feature engineering - Create meaningful features for each component
        # Each component gets a 100-dimensional feature vector representing its characteristics
        for i, component in enumerate(components):
            # Feature 0: Component type encoding (simple hash-based encoding)
            # This helps the model distinguish between different types of services
            features[i, 0] = hash(str(component)) % 10 / 10.0
            
            # Feature 1: Activity level based on log volume
            # Higher log count indicates more active component
            if 'logs' in telemetry_data:
                log_count = len(telemetry_data['logs'][telemetry_data['logs']['cmdb_id'] == component])
                features[i, 1] = min(log_count / 1000.0, 1.0)  # Normalize to [0,1]
            
            # Feature 2: Error rate calculation from log analysis
            # Critical for fault detection - higher error rates indicate problems
            if 'logs' in telemetry_data:
                comp_logs = telemetry_data['logs'][telemetry_data['logs']['cmdb_id'] == component]
                if len(comp_logs) > 0:
                    # Search for error patterns in log messages
                    error_patterns = ['error', 'exception', 'fail', 'timeout']
                    error_count = 0
                    for _, log in comp_logs.iterrows():
                        log_value = str(log.get('value', '')).lower()
                        if any(pattern in log_value for pattern in error_patterns):
                            error_count += 1
                    error_rate = error_count / len(comp_logs)
                    features[i, 2] = error_rate
            
            # Feature 3-10: Metric statistics
            if 'metrics_app' in telemetry_data:
                comp_metrics = telemetry_data['metrics_app'][
                    telemetry_data['metrics_app']['tc'] == component
                ]
                if len(comp_metrics) > 0:
                    # Response rate (rr)
                    if 'rr' in comp_metrics.columns:
                        features[i, 3] = comp_metrics['rr'].mean() / 100.0
                    # Success rate (sr)
                    if 'sr' in comp_metrics.columns:
                        features[i, 4] = comp_metrics['sr'].mean() / 100.0
                    # Mean response time (mrt)
                    if 'mrt' in comp_metrics.columns:
                        features[i, 5] = min(comp_metrics['mrt'].mean() / 1000.0, 1.0)
                    # Count (cnt)
                    if 'cnt' in comp_metrics.columns:
                        features[i, 6] = min(comp_metrics['cnt'].sum() / 10000.0, 1.0)
            
            # Feature 7-9: Container metrics
            if 'metrics_container' in telemetry_data:
                comp_metrics = telemetry_data['metrics_container'][
                    telemetry_data['metrics_container']['cmdb_id'] == component
                ]
                if len(comp_metrics) > 0:
                    # Average metric values
                    if 'value' in comp_metrics.columns:
                        features[i, 7] = min(comp_metrics['value'].mean() / 100.0, 1.0)
                        features[i, 8] = min(comp_metrics['value'].std() / 100.0, 1.0)
                        features[i, 9] = len(comp_metrics) / 1000.0  # Metric count
            
            # Fill remaining features with random values (for now)
            features[i, 10:] = np.random.randn(input_dim - 10) * 0.1
        
        # STEP 4: Create adjacency matrices (3 types representing different relationships)
        # KGroot uses multiple adjacency matrices to capture different types of relationships
        adj_matrices = []
        
        # Adjacency Matrix 1: Service Call Relationships (from distributed traces)
        # This captures the actual service dependency graph - which services call which others
        adj1 = np.zeros((n_components, n_components), dtype=np.float32)
        if 'traces' in telemetry_data:
            traces = telemetry_data['traces']
            # Create relationships based on trace parent-child relationships
            # In distributed tracing, parent_id -> child_id represents a service call
            for _, trace in traces.iterrows():
                if 'parent_id' in trace and 'cmdb_id' in trace and 'trace_id' in trace:
                    parent_id = trace['parent_id']    # Service that made the call
                    child_id = trace['cmdb_id']       # Service that received the call
                    
                    # Find components that match these IDs (fuzzy matching)
                    parent_components = [c for c in components if str(parent_id) in str(c)]
                    child_components = [c for c in components if str(child_id) in str(c)]
                    
                    # Create directed edge from parent to child service
                    for parent_comp in parent_components:
                        for child_comp in child_components:
                            try:
                                parent_idx = components.index(parent_comp)
                                child_idx = components.index(child_comp)
                                adj1[parent_idx, child_idx] = 1.0  # Binary relationship
                            except ValueError:
                                continue
        
        # Adjacency matrix 2: Temporal relationships (components active around same time)
        adj2 = np.zeros((n_components, n_components), dtype=np.float32)
        if 'logs' in telemetry_data:
            logs = telemetry_data['logs']
            for component1 in components:
                for component2 in components:
                    if component1 != component2:
                        comp1_logs = logs[logs['cmdb_id'] == component1]
                        comp2_logs = logs[logs['cmdb_id'] == component2]
                        
                        # Simple temporal correlation based on log timestamps
                        if len(comp1_logs) > 0 and len(comp2_logs) > 0:
                            # Calculate correlation (simplified)
                            time_diff = abs(len(comp1_logs) - len(comp2_logs))
                            adj2[components.index(component1), components.index(component2)] = \
                                max(0, 1.0 - time_diff / max(len(comp1_logs), len(comp2_logs)))
        
        # Adjacency matrix 3: Metric correlation
        adj3 = np.zeros((n_components, n_components), dtype=np.float32)
        if 'metrics_app' in telemetry_data:
            metrics = telemetry_data['metrics_app']
            # Create relationships based on similar metric patterns
            for component1 in components:
                for component2 in components:
                    if component1 != component2:
                        comp1_metrics = metrics[metrics['tc'] == component1]
                        comp2_metrics = metrics[metrics['tc'] == component2]
                        
                        if len(comp1_metrics) > 0 and len(comp2_metrics) > 0:
                            # Calculate correlation based on response times
                            if 'mrt' in comp1_metrics.columns and 'mrt' in comp2_metrics.columns:
                                mrt1 = comp1_metrics['mrt'].mean()
                                mrt2 = comp2_metrics['mrt'].mean()
                                correlation = 1.0 - abs(mrt1 - mrt2) / max(mrt1, mrt2, 1.0)
                                adj3[components.index(component1), components.index(component2)] = \
                                    max(0, correlation)
        
        adj_matrices = [adj1, adj2, adj3]
        
        return features, adj_matrices
    
    def _create_dummy_graph(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Create a dummy graph when no data is available"""
        n_components = 5
        input_dim = 100
        
        features = np.random.randn(n_components, input_dim).astype(np.float32) * 0.1
        adj_matrices = [
            np.eye(n_components, dtype=np.float32),  # Identity
            np.ones((n_components, n_components), dtype=np.float32) * 0.1,  # Weak connections
            np.eye(n_components, dtype=np.float32)  # Identity
        ]
        
        return features, adj_matrices
    
    def _is_fault_period(self, date: str, fault_time: datetime = None) -> bool:
        """Check if the given date contains fault periods"""
        date_obj = datetime.strptime(date, "%Y_%m_%d")
        start_time = date_obj
        end_time = date_obj + timedelta(days=1)
        
        # Check if any fault occurred on this date
        faults_on_date = self.fault_records[
            (self.fault_records['timestamp'] >= start_time) & 
            (self.fault_records['timestamp'] < end_time)
        ]
        
        return len(faults_on_date) > 0
    
    def process_all_dates(self, max_dates: int = 3, sample_size: int = 1000):
        """Process available dates and create pickle files (limited for faster testing)"""
        processed_dates = []
        
        # Get all available dates
        dates = [d for d in os.listdir(self.telemetry_dir) 
                if os.path.isdir(os.path.join(self.telemetry_dir, d))]
        dates.sort()
        
        # Only process first few dates for faster testing
        dates_to_process = dates[:max_dates]
        print(f"Processing {len(dates_to_process)} dates: {dates_to_process}")
        
        for date in dates_to_process:
            print(f"\nProcessing date: {date}")
            
            try:
                # Load telemetry data for this date (with sampling)
                telemetry_data = self._load_telemetry_data(date, sample_size)
                
                # Check if this date has faults
                is_fault_period = self._is_fault_period(date)
                print(f"  Fault period: {is_fault_period}")
                
                # Create graph representation
                features, adj_matrices = self._create_component_graph(telemetry_data)
                
                # Create graph data structure
                graph_data = {
                    'fetures': features,  # Note: keeping the typo as in original code
                    'adj': adj_matrices,
                    'node_index_value': list(range(features.shape[0]))
                }
                
                # Save pickle file
                pickle_path = os.path.join(self.output_dir, "pickle_data", f"{date}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(graph_data, f)
                
                processed_dates.append(date)
                print(f"✅ Saved: {pickle_path}")
                print(f"   Features: {features.shape}, Adj matrices: {len(adj_matrices)}")
                
            except Exception as e:
                print(f"❌ Error processing {date}: {e}")
                continue
        
        return processed_dates
    
    def create_labeled_data(self, processed_dates: List[str]):
        """Create labeled_data.json with positive/negative pairs"""
        labeled_data = []
        
        # Create pairs: each date paired with itself (positive) and others (negative)
        for i, date1 in enumerate(processed_dates):
            for j, date2 in enumerate(processed_dates):
                if i == j:
                    # Same date = positive pair (label = 1)
                    labeled_data.append([f"{date1}.pkl", f"{date2}.pkl", 1])
                else:
                    # Different dates = negative pair (label = 0)
                    labeled_data.append([f"{date1}.pkl", f"{date2}.pkl", 0])
        
        # Save labeled data
        labeled_data_path = os.path.join(self.output_dir, "labeled_data.json")
        with open(labeled_data_path, 'w') as f:
            json.dump(labeled_data, f, indent=2)
        
        print(f"Created labeled_data.json with {len(labeled_data)} pairs")
        return labeled_data_path

def main():
    """Main function to process Bank data"""
    bank_data_dir = "Bank"
    output_dir = "bank_from_bank"
    
    processor = BankDataProcessor(bank_data_dir, output_dir)
    
    print("Processing Bank telemetry data (with sampling for faster processing)...")
    # Only process 3 dates with 1000 samples each for faster testing
    processed_dates = processor.process_all_dates(max_dates=3, sample_size=1000)
    
    print("Creating labeled data...")
    labeled_data_path = processor.create_labeled_data(processed_dates)
    
    print(f"Processing complete! Output saved to: {output_dir}")
    print(f"Processed {len(processed_dates)} dates")

if __name__ == "__main__":
    main()
