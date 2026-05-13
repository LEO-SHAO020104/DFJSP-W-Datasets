import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np


class FJSPWorkerDataGenerator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FJSP with workers dataset generator.

        Args:
            config: Configuration dictionary containing all generation parameters.
        """
        self.n_jobs = config['n_jobs']
        self.n_machines = config['n_machines']
        self.n_workers = config['n_workers']

        # Operation-related parameters.
        self.ops_per_job_min = config['ops_per_job_min']
        self.ops_per_job_max = config['ops_per_job_max']

        # Processing-time parameters.
        self.machine_pt_min = config['machine_pt_min']
        self.machine_pt_max = config['machine_pt_max']
        self.worker_pt_min = config['worker_pt_min']
        self.worker_pt_max = config['worker_pt_max']

        # Compatibility parameters.
        self.min_compatible_machines = config['min_compatible_machines']  # Minimum compatible machines per operation.
        self.max_compatible_machines = config['max_compatible_machines']  # Maximum compatible machines per operation.
        self.min_compatible_workers = config['min_compatible_workers']  # Minimum compatible workers per operation-machine pair.
        self.max_compatible_workers = config['max_compatible_workers']  # Maximum compatible workers per operation-machine pair.

        # Dataset parameters.
        self.num_instances = config['num_instances']
        self.output_dir = config['output_dir']
        self.dataset_name = config['dataset_name']

    def generate_job_lengths(self) -> np.ndarray:
        """Generate the number of operations for each job."""
        job_lengths = np.random.randint(
            self.ops_per_job_min,
            self.ops_per_job_max + 1,
            size=self.n_jobs
        )
        return job_lengths

    def generate_machine_processing_times(self, total_ops: int) -> np.ndarray:
        """
        Generate the operation-machine processing-time matrix.

        Args:
            total_ops: Total number of operations.

        Returns:
            op_pt: Processing-time matrix with shape [total_ops, n_machines].
                A value of 0 means the machine is incompatible with the operation.
        """
        op_pt = np.zeros((total_ops, self.n_machines), dtype=int)

        for i in range(total_ops):
            # Randomly choose the number of compatible machines within the configured range.
            max_compatible = min(self.max_compatible_machines, self.n_machines)
            n_compatible = np.random.randint(self.min_compatible_machines, max_compatible + 1)

            # Randomly choose compatible machines.
            compatible_machines = np.random.choice(
                self.n_machines,
                size=n_compatible,
                replace=False
            )

            # Generate processing times for compatible machines.
            processing_times = np.random.randint(
                self.machine_pt_min,
                self.machine_pt_max + 1,
                size=n_compatible
            )

            op_pt[i, compatible_machines] = processing_times

        return op_pt

    def generate_worker_processing_times(self, total_ops: int, op_pt: np.ndarray) -> np.ndarray:
        """
        Generate the three-dimensional operation-machine-worker processing-time matrix.

        Args:
            total_ops: Total number of operations.
            op_pt: Operation-machine processing-time matrix used to identify available machines.

        Returns:
            op_wt: Processing-time matrix with shape [total_ops, n_machines, n_workers].
                A value of 0 means the worker is incompatible with the operation-machine pair.
                op_wt[i, m, w] is the processing time of operation i on machine m with worker w.
        """
        op_wt = np.zeros((total_ops, self.n_machines, self.n_workers), dtype=int)

        for i in range(total_ops):
            # Find machines that can process operation i.
            available_machines = np.where(op_pt[i] > 0)[0]

            for m in available_machines:
                # Generate worker compatibility for each available machine.
                max_compatible = min(self.max_compatible_workers, self.n_workers)
                n_compatible = np.random.randint(self.min_compatible_workers, max_compatible + 1)

                # Randomly choose compatible workers.
                compatible_workers = np.random.choice(
                    self.n_workers,
                    size=n_compatible,
                    replace=False
                )

                # Generate processing times for compatible workers.
                processing_times = np.random.randint(
                    self.worker_pt_min,
                    self.worker_pt_max + 1,
                    size=n_compatible
                )

                op_wt[i, m, compatible_workers] = processing_times

        return op_wt

    def generate_single_instance(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate one FJSP with workers instance using a three-dimensional worker processing-time matrix.

        Returns:
            job_lengths: Number of operations for each job, shape [n_jobs].
            op_pt: Operation-machine processing-time matrix, shape [total_ops, n_machines].
            op_wt: Operation-machine-worker processing-time matrix, shape [total_ops, n_machines, n_workers].
        """
        # Generate the number of operations for each job.
        job_lengths = self.generate_job_lengths()
        total_ops = np.sum(job_lengths)

        # Generate the machine processing-time matrix first.
        op_pt = self.generate_machine_processing_times(total_ops)

        # Generate the worker processing-time matrix based on machine compatibility.
        op_wt = self.generate_worker_processing_times(total_ops, op_pt)

        return job_lengths, op_pt, op_wt

    def validate_instance(self, job_lengths: np.ndarray, op_pt: np.ndarray, op_wt: np.ndarray) -> bool:
        """
        Validate a generated instance.

        Args:
            job_lengths: Number of operations for each job.
            op_pt: Operation-machine processing-time matrix.
            op_wt: Operation-machine-worker processing-time matrix.

        Returns:
            bool: True if the instance is valid, otherwise False.
        """
        total_ops = np.sum(job_lengths)

        # Check matrix shapes.
        if op_pt.shape != (total_ops, self.n_machines):
            return False
        if op_wt.shape != (total_ops, self.n_machines, self.n_workers):
            return False

        # Check that every operation has at least one available machine-worker combination.
        for i in range(total_ops):
            available_machines = np.where(op_pt[i] > 0)[0]
            if len(available_machines) == 0:  # No compatible machine.
                return False

            # Check that each available machine has at least one compatible worker.
            has_valid_combination = False
            for m in available_machines:
                if np.sum(op_wt[i, m] > 0) > 0:  # This machine has at least one compatible worker.
                    has_valid_combination = True
                    break

            if not has_valid_combination:
                return False

        return True

    def generate_dataset(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate the full dataset.

        Returns:
            dataset: List of all instances. Each instance is a tuple of (job_lengths, op_pt, op_wt).
        """
        dataset = []
        valid_instances = 0
        attempts = 0
        max_attempts = self.num_instances * 10  # Maximum number of generation attempts.

        print(f"Generating {self.num_instances} instances...")

        while valid_instances < self.num_instances and attempts < max_attempts:
            attempts += 1

            try:
                job_lengths, op_pt, op_wt = self.generate_single_instance()

                if self.validate_instance(job_lengths, op_pt, op_wt):
                    dataset.append((job_lengths, op_pt, op_wt))
                    valid_instances += 1

                    if valid_instances % 10 == 0:
                        print(f"Generated {valid_instances}/{self.num_instances} valid instances")

            except Exception as e:
                print(f"Error while generating an instance: {e}")
                continue

        if valid_instances < self.num_instances:
            print(f"Warning: only {valid_instances}/{self.num_instances} instances were generated successfully")

        return dataset

    def save_dataset(self, dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        """
        Save the dataset to files. Each instance is stored as a separate pkl file.

        Args:
            dataset: Dataset to save. Each element is a tuple of (job_lengths, op_pt, op_wt).
        """
        # Create the output directory.
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Saving {len(dataset)} instances as separate files...")

        # Save each instance as a separate pkl file.
        saved_files = []
        for i, (job_lengths, op_pt, op_wt) in enumerate(dataset, 1):
            filename = f"{self.dataset_name}_{i:03d}.pkl"
            filepath = os.path.join(self.output_dir, filename)

            # Save one instance.
            instance_data = {
                'job_lengths': job_lengths,
                'op_pt': op_pt,
                'op_wt': op_wt,
                'instance_info': {
                    'n_jobs': self.n_jobs,
                    'n_machines': self.n_machines,
                    'n_workers': self.n_workers,
                    'total_operations': np.sum(job_lengths)
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(instance_data, f)

            saved_files.append(filename)

            if i % 20 == 0:
                print(f"Saved {i}/{len(dataset)} instances")

        # Create an index file for batch loading.
        index_file = os.path.join(self.output_dir, f"{self.dataset_name}_index.json")
        index_data = {
            'dataset_name': self.dataset_name,
            'total_instances': len(dataset),
            'files': saved_files,
            'config': {
                'n_jobs': self.n_jobs,
                'n_machines': self.n_machines,
                'n_workers': self.n_workers,
                'ops_per_job_range': [self.ops_per_job_min, self.ops_per_job_max],
                'machine_pt_range': [self.machine_pt_min, self.machine_pt_max],
                'worker_pt_range': [self.worker_pt_min, self.worker_pt_max],
                'min_compatible_machines': self.min_compatible_machines,
                'max_compatible_machines': self.max_compatible_machines,
                'min_compatible_workers': self.min_compatible_workers,
                'max_compatible_workers': self.max_compatible_workers,
                'modeling_approach': 'three_dimensional_worker_processing_time'
            }
        }

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)

        print(f"\nDataset saved to: {self.output_dir}")
        print(f"  Instance files: {self.dataset_name}_001.pkl ~ {self.dataset_name}_{len(dataset):03d}.pkl")
        print(f"  Index file: {self.dataset_name}_index.json")
        print("\nEach instance file contains:")
        print("  - job_lengths: array containing the number of operations for each job")
        print("  - op_pt: [ops, machines] operation-machine processing-time matrix")
        print("  - op_wt: [ops, machines, workers] operation-machine-worker processing-time matrix")
        print("  - instance_info: basic instance metadata")

    def print_dataset_statistics(self, dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        """Print dataset statistics."""
        if not dataset:
            print("The dataset is empty")
            return

        total_ops_list = [np.sum(job_lengths) for job_lengths, _, _ in dataset]
        machine_compatible_ops = []
        worker_compatible_combinations = []

        for _, op_pt, op_wt in dataset:
            # Count machine compatibility.
            machine_compatible_ops.extend([np.sum(row > 0) for row in op_pt])

            # Count valid operation-machine-worker combinations.
            total_ops = op_pt.shape[0]
            for i in range(total_ops):
                valid_combinations = 0
                available_machines = np.where(op_pt[i] > 0)[0]
                for m in available_machines:
                    valid_combinations += np.sum(op_wt[i, m] > 0)
                worker_compatible_combinations.append(valid_combinations)

        print("\n=== Dataset Statistics ===")
        print(f"Number of instances: {len(dataset)}")
        print(f"Number of jobs: {self.n_jobs}")
        print(f"Number of machines: {self.n_machines}")
        print(f"Number of workers: {self.n_workers}")
        print(f"Total operations range: {min(total_ops_list)} - {max(total_ops_list)}")
        print(f"Average total operations: {np.mean(total_ops_list):.1f}")
        print(f"Compatible machines per operation range: {min(machine_compatible_ops)} - {max(machine_compatible_ops)}")
        print(f"Average compatible machines per operation: {np.mean(machine_compatible_ops):.1f}")
        print(f"Available machine-worker combinations per operation range: {min(worker_compatible_combinations)} - {max(worker_compatible_combinations)}")
        print(f"Average available machine-worker combinations per operation: {np.mean(worker_compatible_combinations):.1f}")
        print("\nModeling approach: three-dimensional worker processing-time matrix")
        print("  - op_wt[i, m, w] = processing time of operation i on machine m with worker w")
        print("  - Each operation requires selecting one machine-worker combination")
        print("  - Worker processing time depends on the selected machine")


def load_dataset(dataset_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Load a dataset from a single instance file, an index file, or a dataset directory.

    Args:
        dataset_path: Dataset path.
            - Single instance: "dataset_name_001.pkl"
            - Full dataset: "dataset_name_index.json" or a dataset directory

    Returns:
        job_lengths_list, op_pt_list, op_wt_list
    """
    if dataset_path.endswith('.pkl'):
        # Load a single instance.
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
            return [data['job_lengths']], [data['op_pt']], [data['op_wt']]

    elif dataset_path.endswith('_index.json'):
        # Load the full dataset through the index file.
        with open(dataset_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        dataset_dir = os.path.dirname(dataset_path)
        job_lengths_list, op_pt_list, op_wt_list = [], [], []

        print(f"Loading dataset: {index_data['dataset_name']}")
        print(f"Total instances: {index_data['total_instances']}")

        for i, filename in enumerate(index_data['files'], 1):
            filepath = os.path.join(dataset_dir, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                job_lengths_list.append(data['job_lengths'])
                op_pt_list.append(data['op_pt'])
                op_wt_list.append(data['op_wt'])

            if i % 20 == 0:
                print(f"Loaded {i}/{len(index_data['files'])} instances")

        return job_lengths_list, op_pt_list, op_wt_list

    elif os.path.isdir(dataset_path):
        # Load all pkl files from a directory.
        pkl_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.pkl') and '_index.json' not in f])

        if not pkl_files:
            raise ValueError(f"No pkl files were found in directory: {dataset_path}")

        job_lengths_list, op_pt_list, op_wt_list = [], [], []

        print(f"Loading dataset from directory: {dataset_path}")
        print(f"Found {len(pkl_files)} instance files")

        for i, filename in enumerate(pkl_files, 1):
            filepath = os.path.join(dataset_path, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                job_lengths_list.append(data['job_lengths'])
                op_pt_list.append(data['op_pt'])
                op_wt_list.append(data['op_wt'])

            if i % 20 == 0:
                print(f"Loaded {i}/{len(pkl_files)} instances")

        return job_lengths_list, op_pt_list, op_wt_list

    else:
        raise ValueError("Unsupported path format. Use a pkl file, an index.json file, or a dataset directory.")


def load_single_instance(instance_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load a single instance.

    Args:
        instance_path: Instance file path (.pkl).

    Returns:
        job_lengths, op_pt, op_wt, instance_info
    """
    with open(instance_path, 'rb') as f:
        data = pickle.load(f)

    return data['job_lengths'], data['op_pt'], data['op_wt'], data['instance_info']


def load_dataset_config(index_path: str) -> Dict:
    """
    Load dataset configuration information.

    Args:
        index_path: Index file path (_index.json).

    Returns:
        config: Dataset configuration information.
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    return index_data['config']


def main():
    parser = argparse.ArgumentParser(description='Generate FJSP with workers datasets using a three-dimensional worker processing-time matrix.')

    # Basic parameters.
    parser.add_argument('--n_jobs', type=int, default=50, help='Number of jobs')
    parser.add_argument('--n_machines', type=int, default=10, help='Number of machines')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers')

    # Operation parameters.
    parser.add_argument('--ops_per_job_min', type=int, default=5, help='Minimum number of operations per job')
    parser.add_argument('--ops_per_job_max', type=int, default=5, help='Maximum number of operations per job')

    # Processing-time parameters.
    parser.add_argument('--machine_pt_min', type=int, default=1, help='Lower bound of machine processing time')
    parser.add_argument('--machine_pt_max', type=int, default=50, help='Upper bound of machine processing time')
    parser.add_argument('--worker_pt_min', type=int, default=1, help='Lower bound of worker processing time')
    parser.add_argument('--worker_pt_max', type=int, default=50, help='Upper bound of worker processing time')

    # Compatibility parameters.
    parser.add_argument('--min_compatible_machines', type=int, default=1, help='Minimum compatible machines per operation')
    parser.add_argument('--max_compatible_machines', type=int, default=10, help='Maximum compatible machines per operation')
    parser.add_argument('--min_compatible_workers', type=int, default=1, help='Minimum compatible workers per operation-machine pair')
    parser.add_argument('--max_compatible_workers', type=int, default=5, help='Maximum compatible workers per operation-machine pair')

    # Dataset parameters.
    parser.add_argument('--num_instances', type=int, default=100, help='Number of instances to generate')
    parser.add_argument('--output_dir', type=str, default='valid/50x10x8+mix', help='Output directory')
    parser.add_argument('--dataset_name', type=str, default='50x10x8', help='Dataset name')

    args = parser.parse_args()

    # Build the configuration dictionary.
    config = vars(args)

    # Create the generator and generate the dataset.
    generator = FJSPWorkerDataGenerator(config)
    dataset = generator.generate_dataset()

    # Save the dataset.
    generator.save_dataset(dataset)

    # Print statistics.
    generator.print_dataset_statistics(dataset)


if __name__ == "__main__":
    main()
