"""
Comprehensive benchmarking suite for Agent-RLlib
Measures training speed, inference latency, and memory usage across different configurations.
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List
import argparse
import json
from pathlib import Path

from src.agent_rllib.envs import SupportBotEnv
from src.agent_rllib.agents import HybridPPOAgent


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for Agent-RLlib."""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.results = {}
        
    def benchmark_environment(self, num_episodes: int = 100) -> Dict:
        """Benchmark environment reset and step operations."""
        print("🌍 Benchmarking environment performance...")
        
        env = SupportBotEnv(difficulty="medium", max_turns=20)
        
        # Benchmark resets
        reset_times = []
        for _ in range(num_episodes):
            start = time.perf_counter()
            env.reset()
            reset_times.append(time.perf_counter() - start)
        
        # Benchmark steps
        step_times = []
        obs, _ = env.reset()
        for _ in range(num_episodes * 10):  # More steps than episodes
            dummy_action = {
                "tool_id": 0,
                "response_type": 0, 
                "confidence": np.array([0.5])
            }
            start = time.perf_counter()
            obs, _, done, _, _ = env.step(dummy_action)
            step_times.append(time.perf_counter() - start)
            
            if done:
                obs, _ = env.reset()
        
        return {
            "reset_time_mean": np.mean(reset_times) * 1000,  # ms
            "reset_time_std": np.std(reset_times) * 1000,
            "step_time_mean": np.mean(step_times) * 1000,
            "step_time_std": np.std(step_times) * 1000,
            "episodes_per_second": 1.0 / np.mean(reset_times),
            "steps_per_second": 1.0 / np.mean(step_times)
        }
    
    def benchmark_agent_inference(self, num_inferences: int = 1000) -> Dict:
        """Benchmark agent inference speed."""
        print("🤖 Benchmarking agent inference...")
        
        env = SupportBotEnv(difficulty="medium")
        agent = HybridPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        # Warm up
        obs, _ = env.reset()
        for _ in range(10):
            agent.get_action(obs)
        
        # Benchmark inference
        inference_times = []
        memory_usage = []
        
        for _ in range(num_inferences):
            obs, _ = env.reset()
            
            # Memory before
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
            else:
                mem_before = psutil.Process().memory_info().rss
            
            # Inference timing
            start = time.perf_counter()
            action = agent.get_action(obs, deterministic=True)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            inference_time = time.perf_counter() - start
            inference_times.append(inference_time)
            
            # Memory after
            if self.device.type == "cuda":
                mem_after = torch.cuda.memory_allocated()
                memory_usage.append((mem_after - mem_before) / 1024**2)  # MB
            else:
                mem_after = psutil.Process().memory_info().rss
                memory_usage.append((mem_after - mem_before) / 1024**2)
        
        return {
            "inference_time_mean": np.mean(inference_times) * 1000,  # ms
            "inference_time_std": np.std(inference_times) * 1000,
            "inference_time_p95": np.percentile(inference_times, 95) * 1000,
            "inference_time_p99": np.percentile(inference_times, 99) * 1000,
            "inferences_per_second": 1.0 / np.mean(inference_times),
            "memory_usage_mean": np.mean(memory_usage),
            "memory_usage_max": np.max(memory_usage),
            "device": str(self.device)
        }
    
    def benchmark_training_throughput(self, num_steps: int = 1000) -> Dict:
        """Benchmark training throughput."""
        print("🏋️ Benchmarking training throughput...")
        
        env = SupportBotEnv(difficulty="medium")
        agent = HybridPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        # Collect experience
        start_time = time.perf_counter()
        total_reward = 0
        episodes_completed = 0
        
        obs, _ = env.reset()
        
        for step in range(num_steps):
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            agent.update(obs, reward, done or truncated, info)
            
            if done or truncated:
                episodes_completed += 1
                obs, _ = env.reset()
        
        total_time = time.perf_counter() - start_time
        
        return {
            "total_time": total_time,
            "steps_per_second": num_steps / total_time,
            "episodes_completed": episodes_completed,
            "average_episode_reward": total_reward / max(episodes_completed, 1),
            "training_efficiency": num_steps / total_time / 1000  # k-steps/sec
        }
    
    def benchmark_memory_scaling(self, batch_sizes: List[int] = None) -> Dict:
        """Benchmark memory usage with different batch sizes."""
        print("💾 Benchmarking memory scaling...")
        
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64]
        
        env = SupportBotEnv(difficulty="medium")
        results = {}
        
        for batch_size in batch_sizes:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            agent = HybridPPOAgent(
                observation_space=env.observation_space,
                action_space=env.action_space
            )
            
            # Simulate batch processing
            obs_batch = []
            for _ in range(batch_size):
                obs, _ = env.reset()
                obs_batch.append(obs)
            
            # Memory measurement
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
            
            # Process batch
            for obs in obs_batch:
                agent.get_action(obs)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                
                results[batch_size] = {
                    "allocated_memory": (mem_after - mem_before) / 1024**2,  # MB
                    "peak_memory": peak_memory / 1024**2,
                    "memory_per_sample": (mem_after - mem_before) / batch_size / 1024**2
                }
            else:
                # CPU memory estimation
                process = psutil.Process()
                mem_info = process.memory_info()
                results[batch_size] = {
                    "rss_memory": mem_info.rss / 1024**2,
                    "vms_memory": mem_info.vms / 1024**2
                }
            
            del agent  # Clean up
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run comprehensive benchmark suite."""
        print("🚀 Running comprehensive Agent-RLlib benchmark...")
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        results = {
            "system_info": {
                "device": str(self.device),
                "pytorch_version": torch.__version__,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total / 1024**3,  # GB
            }
        }
        
        if self.device.type == "cuda":
            results["system_info"].update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        # Run benchmarks
        results["environment"] = self.benchmark_environment()
        results["inference"] = self.benchmark_agent_inference()
        results["training"] = self.benchmark_training_throughput()
        results["memory_scaling"] = self.benchmark_memory_scaling()
        
        return results
    
    def save_results(self, results: Dict, filepath: str):
        """Save benchmark results to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📊 Results saved to {filepath}")
    
    def print_summary(self, results: Dict):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("🏆 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n🌍 Environment Performance:")
        env_results = results["environment"]
        print(f"  Reset time: {env_results['reset_time_mean']:.2f}ms ± {env_results['reset_time_std']:.2f}ms")
        print(f"  Step time: {env_results['step_time_mean']:.2f}ms ± {env_results['step_time_std']:.2f}ms")
        print(f"  Episodes/sec: {env_results['episodes_per_second']:.1f}")
        print(f"  Steps/sec: {env_results['steps_per_second']:.1f}")
        
        print(f"\n🤖 Agent Inference:")
        inf_results = results["inference"]
        print(f"  Mean latency: {inf_results['inference_time_mean']:.2f}ms")
        print(f"  P95 latency: {inf_results['inference_time_p95']:.2f}ms")
        print(f"  P99 latency: {inf_results['inference_time_p99']:.2f}ms")
        print(f"  Throughput: {inf_results['inferences_per_second']:.1f} inferences/sec")
        print(f"  Memory usage: {inf_results['memory_usage_mean']:.2f}MB (avg)")
        
        print(f"\n🏋️ Training Performance:")
        train_results = results["training"]
        print(f"  Training speed: {train_results['steps_per_second']:.1f} steps/sec")
        print(f"  Episodes completed: {train_results['episodes_completed']}")
        print(f"  Average reward: {train_results['average_episode_reward']:.2f}")
        
        print(f"\n💾 Memory Scaling:")
        mem_results = results["memory_scaling"]
        for batch_size, mem_info in mem_results.items():
            if "allocated_memory" in mem_info:
                print(f"  Batch {batch_size}: {mem_info['allocated_memory']:.1f}MB allocated, {mem_info['peak_memory']:.1f}MB peak")


def main():
    parser = argparse.ArgumentParser(description="Agent-RLlib Performance Benchmark")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="Device to run benchmark on")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with reduced iterations")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(device=args.device)
    
    # Adjust iterations for quick mode
    if args.quick:
        print("🚀 Running quick benchmark...")
        # Override with smaller numbers for quick testing
        benchmark.benchmark_environment = lambda: benchmark.benchmark_environment(num_episodes=10)
        benchmark.benchmark_agent_inference = lambda: benchmark.benchmark_agent_inference(num_inferences=100)
        benchmark.benchmark_training_throughput = lambda: benchmark.benchmark_training_throughput(num_steps=100)
    
    # Run benchmark
    results = benchmark.run_full_benchmark()
    
    # Save and display results
    benchmark.save_results(results, args.output)
    benchmark.print_summary(results)
    
    print(f"\n✅ Benchmark completed! Results saved to {args.output}")


if __name__ == "__main__":
    main()
