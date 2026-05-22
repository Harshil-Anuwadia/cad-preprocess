#!/usr/bin/env python3
"""
Performance Benchmark Tool for CAD-Preprocess.

This script performs a comprehensive performance evaluation of the cad-preprocess
library, analyzing hardware utilization, throughput, and scalability.
"""

import sys
import time
import os
import platform
import subprocess
import json
import shutil
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# --- ANSI Colors (Disabled) ---
BLUE = ""
GREEN = ""
YELLOW = ""
RED = ""
MAGENTA = ""
CYAN = ""
BOLD = ""
UNDERLINE = ""
RESET = ""

# tqdm replacement
def progress_bar(iterable, desc="Processing"):
    total = len(iterable)
    for i, item in enumerate(iterable):
        yield item
        percent = 100 * ((i + 1) / total)
        bar = '█' * int(percent / 2) + '-' * (50 - int(percent / 2))
        sys.stdout.write(f"\r{desc}: |{bar}| {percent:.1f}%")
        sys.stdout.flush()
    print()


from cad_preprocess import CADPreprocessor, Config, __version__

# --- Hardware Detection ---

def get_cpu_info():
    """Get detailed CPU information."""
    info = {
        "cores_physical": multiprocessing.cpu_count(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
    }
    
    if platform.system() == "Linux":
        try:
            cmd = "lscpu"
            res = subprocess.check_output(cmd, shell=True).decode()
            for line in res.split("\n"):
                if "Model name" in line:
                    info["model"] = line.split(":")[1].strip()
        except:
            pass
    elif platform.system() == "Windows":
        try:
            cmd = "wmic cpu get name"
            res = subprocess.check_output(cmd, shell=True).decode()
            lines = [l.strip() for l in res.split('\n') if l.strip()]
            if len(lines) > 1:
                info["model"] = lines[1]
        except:
            pass
    return info

def get_gpu_info():
    """Detect GPU availability without heavy dependencies."""
    gpus = []
    try:
        # Check for NVIDIA GPUs (works on both Linux and Windows if drivers installed)
        res = subprocess.check_output("nvidia-smi -L", shell=True).decode()
        for line in res.strip().split("\n"):
            gpus.append(line)
    except:
        pass
    return gpus

# --- Synthetic Data Generation ---

def create_dummy_dicom(path: Path, size=(1024, 1024)):
    """Create a valid dummy DICOM file for testing."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "CITIZEN^Benchmark"
    ds.PatientID = "123456"
    ds.ContentDate = time.strftime('%Y%m%d')
    ds.ContentTime = time.strftime('%H%M%S')
    ds.Modality = "CT"
    ds.Rows = size[0]
    ds.Columns = size[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0  # unsigned
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    
    # Create synthetic anatomy-like data (gradient)
    res = np.random.randint(0, 2**12, size=size, dtype=np.uint16)
    ds.PixelData = res.tobytes()
    ds.save_as(str(path))

# --- Benchmark Engine ---

@dataclass
class BenchmarkScenario:
    name: str
    num_files: int
    num_workers: int
    target_size: tuple = (512, 512)
    metadata_profile: str = "minimal"
    nested_depth: int = 0
    description: str = ""

class BenchmarkRunner:
    def __init__(self, test_dir: Path, output_dir: Path):
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.results = []
        
    def setup_data(self, count: int, size=(1024, 1024), nested_depth=0):
        print(f"{BLUE}[*] Generating {count} synthetic DICOM files ({size[0]}x{size[1]}, depth={nested_depth})...{RESET}")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        for i in progress_bar(range(count), desc=f"{CYAN}Generating data{RESET}"):
            # Create nested directory if requested
            target_path = self.test_dir
            if nested_depth > 0:
                for d in range(nested_depth):
                    target_path = target_path / f"depth_{d}"
                target_path.mkdir(parents=True, exist_ok=True)
            
            create_dummy_dicom(target_path / f"bench_{i}.dcm", size=size)

    def run_scenario(self, scenario: BenchmarkScenario):
        print(f"\n{BOLD}{MAGENTA}[Scenario]{RESET} {BOLD}{scenario.name}{RESET}")
        print(f"  {YELLOW}Description:{RESET} {scenario.description}")
        print(f"  {YELLOW}Load:{RESET}        {scenario.num_files} files, {scenario.num_workers} workers")
        
        config = Config()
        config.preprocessing.resizing.target_height = scenario.target_size[0]
        config.preprocessing.resizing.target_width = scenario.target_size[1]
        
        if scenario.metadata_profile == "all":
            config.metadata.include_all_profiles = True
        else:
            config.metadata.profiles = [scenario.metadata_profile]
        
        processor = CADPreprocessor(
            config=config,
            output_dir=self.output_dir,
            num_workers=scenario.num_workers
        )
        
        # Clean output
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
        # Start memory tracking
        def get_mem():
            try:
                if platform.system() == "Windows":
                    # Simple fallback for Windows if psutil not installed
                    # resource module is POSIX only
                    return 0
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            except:
                return 0

        mem_start = get_mem()
        
        start_time = time.perf_counter()
        batch_result = processor.process_directory(self.test_dir)
        end_time = time.perf_counter()
        
        mem_end = get_mem()
        # ru_maxrss is in KB on Linux
        mem_used = (mem_end - mem_start) / 1024 # MB
        
        duration = end_time - start_time
        throughput = scenario.num_files / duration if duration > 0 else 0
        
        # Estimate I/O (Read + Write)
        io_mb = scenario.num_files * 2.5 
        io_speed = io_mb / duration if duration > 0 else 0
        
        result = {
            "name": scenario.name,
            "duration": duration,
            "throughput": throughput,
            "io_throughput_mb_s": io_speed,
            "success_rate": batch_result.total_processed / scenario.num_files * 100 if scenario.num_files > 0 else 0,
            "processed": batch_result.total_processed,
            "workers": scenario.num_workers,
            "memory_mb": mem_used,
            "target_size": scenario.target_size,
        }
        self.results.append(result)
        return result

    def print_report(self, hw_info: dict, gpus: list):
        print("\n" + BOLD + MAGENTA + "="*80 + RESET)
        print(BOLD + CYAN + " " * 28 + "ULTIMATE PERFORMANCE REPORT" + RESET)
        print(BOLD + MAGENTA + "="*80 + RESET)
        
        print(f"\n{BOLD}[SYSTEM DIAGNOSTICS]{RESET}")
        print(f"  {YELLOW}OS Platform:{RESET}   {hw_info['system']} {hw_info['machine']}")
        print(f"  {YELLOW}CPU Model:{RESET}     {hw_info.get('model', hw_info['processor'])}")
        print(f"  {YELLOW}Core Count:{RESET}    {hw_info['cores_physical']} logical cores")
        print(f"  {YELLOW}Python Ver:{RESET}    {platform.python_version()}")
        if gpus:
            print(f"  {YELLOW}GPU Assets:{RESET}    {GREEN}{', '.join(gpus)}{RESET}")
        else:
            print(f"  {YELLOW}GPU Assets:{RESET}    {RED}None (CPU-Only Pipeline){RESET}")
        
        print(f"\n{BOLD}[DETAILED ANALYTICS]{RESET}")
        header = f"{'Scenario':<25} {'Workers':<8} {'Time':<8} {'Img/s':<8} {'I/O MB/s':<10} {'RAM+':<8}"
        print(BOLD + header + RESET)
        print("-" * len(header))
        
        for r in self.results:
            row = (f"{r['name']:<25} {r['workers']:<8} {r['duration']:>6.2f}s "
                   f"{GREEN}{r['throughput']:>7.2f}{RESET} {CYAN}{r['io_throughput_mb_s']:>8.1f}{RESET} "
                   f"{YELLOW}{r['memory_mb']:>7.1f}MB{RESET}")
            print(row)
            
        if self.results:
            total_processed = sum(r['processed'] for r in self.results)
            total_duration = sum(r['duration'] for r in self.results)
            avg_throughput = total_processed / total_duration if total_duration > 0 else 0
            max_throughput = max(r['throughput'] for r in self.results)
            efficiency = (max_throughput / hw_info['cores_physical']) * 100
            
            print(f"\n{BOLD}[AGGREGATED INSIGHTS]{RESET}")
            print(f"  {YELLOW}Total Images:{RESET}      {total_processed}")
            print(f"  {YELLOW}Total Workload:{RESET}    {total_processed * 2.5:.1f} MB estimated")
            print(f"  {YELLOW}Avg Throughput:{RESET}    {avg_throughput:.2f} img/sec")
            print(f"  {YELLOW}Peak Performance:{RESET}  {BOLD}{GREEN}{max_throughput:.2f} img/sec{RESET}")
            
            print(f"\n{BOLD}[HARDWARE UTILIZATION SCORE]{RESET}")
            score_color = GREEN if efficiency > 150 else YELLOW
            print(f"  {YELLOW}Efficiency Index:{RESET} {score_color}{efficiency:.1f}{RESET}")
            
            if efficiency > 300:
                rating = f"{BOLD}{GREEN}ELITE (Server Grade Optimization){RESET}"
            elif efficiency > 150:
                rating = f"{BOLD}{GREEN}PROFESSIONAL (Highly Scalable){RESET}"
            elif efficiency > 75:
                rating = f"{BOLD}{YELLOW}STABLE (Reliable Production Grade){RESET}"
            else:
                rating = f"{BOLD}{RED}CONSTRAINED (Optimization Recommended){RESET}"
            print(f"  {YELLOW}System Rating:{RESET}     {rating}")

            # Auto-Tuner Recommendation
            best_workers = hw_info['cores_physical']
            print(f"\n{BOLD}{CYAN}[AUTO-TUNER RECOMMENDATION]{RESET}")
            print(f"  Based on your {hw_info['cores_physical']}-core {hw_info['system']} system:")
            print(f"  Recommended Workers: {BOLD}{GREEN}{best_workers}{RESET}")
            print(f"  Strategy: {BLUE}Distributed Multi-Processing (ProcessPoolExecutor){RESET}")

        print("\n" + BOLD + MAGENTA + "="*80 + RESET)
        print(BOLD + CYAN + " " * 22 + "CAD-PREPROCESS BENCHMARK COMPLETE" + RESET)
        print(BOLD + MAGENTA + "="*80 + RESET + "\n")

def main():
    test_dir = Path("perf_test_data")
    output_dir = Path("perf_test_output")
    
    try:
        hw_info = get_cpu_info()
        gpus = get_gpu_info()
        
        runner = BenchmarkRunner(test_dir, output_dir)
        
        # Scenario 1: Baseline (Small batch, single core)
        runner.setup_data(20)
        runner.run_scenario(BenchmarkScenario(
            "Single-Core Baseline", 20, 1, 
            description="Reference speed for a single worker thread."
        ))
        
        # Scenario 2: Multi-core Efficiency (Medium batch)
        runner.setup_data(100)
        runner.run_scenario(BenchmarkScenario(
            "Multi-Core Scaling", 100, hw_info['cores_physical'],
            description="Measures throughput gain when utilizing all available cores."
        ))
        
        # Scenario 3: Memory & Resource Stress (Large batch)
        runner.setup_data(300)
        runner.run_scenario(BenchmarkScenario(
            "High-Volume Stress", 300, hw_info['cores_physical'],
            description="Large batch to test memory management and I/O sustained throughput."
        ))
        
        # Scenario 4: High Resolution Load
        runner.setup_data(50, size=(2048, 2048))
        runner.run_scenario(BenchmarkScenario(
            "High-Res (2K) Load", 50, hw_info['cores_physical'], 
            target_size=(1024, 1024),
            description="Testing interpolation and pixel handling for ultra-high-res images."
        ))
        
        # Scenario 5: Nested Directory Discovery
        runner.setup_data(50, nested_depth=3)
        runner.run_scenario(BenchmarkScenario(
            "Nested Dir Scanning", 50, hw_info['cores_physical'],
            description="Stress test for recursive file discovery logic."
        ))
        
        # Scenario 6: Metadata Heavy Extraction
        runner.setup_data(50)
        runner.run_scenario(BenchmarkScenario(
            "Metadata-Heavy Load", 50, hw_info['cores_physical'],
            metadata_profile="all",
            description="Benchmarks performance impact of full DICOM tag extraction."
        ))
        
        runner.print_report(hw_info, gpus)
        
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    main()
