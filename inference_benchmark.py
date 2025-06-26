import asyncio
import argparse
import os
import time
import statistics
import random
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
#from tinfoil import AsyncTinfoilAI

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

def load_realistic_prompts(dataset_name: str = "databricks/databricks-dolly-15k", max_prompts: int = 100):
    """Load diverse prompts from Hugging Face dataset"""
    if not DATASETS_AVAILABLE:
        print("⚠️  datasets library not available. Install with: pip install datasets")
        return None
    
    try:
        print(f"📥 Loading prompts from {dataset_name}...")
        dataset = load_dataset(dataset_name, split="train")
        
        # Extract prompts from dolly dataset (instruction + context if available)
        prompts = []
        for i, item in enumerate(dataset.select(range(min(max_prompts, len(dataset))))):
            if 'instruction' in item:
                prompt = item['instruction']
                # Add context if available and not empty
                if 'context' in item and item['context'].strip():
                    prompt = f"{prompt}\n\nContext: {item['context']}"
                prompts.append(prompt)
        
        # Shuffle for randomness
        random.shuffle(prompts)
        
        print(f"✅ Loaded {len(prompts)} diverse prompts")
        print(f"   Sample lengths: {[len(p) for p in prompts[:5]]} characters")
        return prompts
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None

def get_prompt_for_request(prompts: list, request_id: int, single_prompt: str = None):
    """Get appropriate prompt for a request"""
    if prompts and len(prompts) > 0:
        # Cycle through prompts if we have more requests than prompts
        return prompts[request_id % len(prompts)]
    else:
        return single_prompt

async def chat_integration(prompt: str, client: AsyncOpenAI, tokens_per_second: dict, requests_active_per_second: dict, request_id: int = None):

	if request_id is not None:
		print(f"\n--- Request {request_id} ---")
		print(f"Prompt length: {len(prompt)} chars")
	
	start_time = time.time()
	first_token_time = None
	
	stream = await client.chat.completions.create(
	    messages=[
		{
		    "role": "user",
		    "content": prompt,
		}
	    ],
		# OpenAI
		# model="gpt-4o",

		# Tinfoil AI
	    model="qwen2-5-72b",
		# model="deepseek-r1-70b",
	    stream=True,
	)

	async for chunk in stream:
		if chunk.choices[0].delta.content is not None:
			if first_token_time is None:
				first_token_time = time.time()
			
			# Count tokens and track per second (each chunk = 1 token)
			current_time = time.time()
			second_key = int(current_time)
			
			if chunk.choices[0].delta.content:
				tokens_per_second[second_key] += 1  # Each chunk = 1 token
				requests_active_per_second[second_key].add(request_id)
			
			#print(chunk.choices[0].delta.content, end="", flush=True)
	#print()
	
	if request_id is not None:
		ttft = first_token_time - start_time if first_token_time else None
		if ttft:
			print(f"--- End Request {request_id} (TTFT: {ttft:.3f}s) ---\n")
		else:
			print(f"--- End Request {request_id} (No tokens received) ---\n")
		return ttft
	
	return None

def analyze_throughput(tokens_per_second: dict, requests_active_per_second: dict):
	"""Analyze throughput metrics from time-series data"""
	if not tokens_per_second:
		print("No throughput data collected.")
		return
	
	# Find peak throughput
	peak_second = max(tokens_per_second, key=tokens_per_second.get)
	peak_throughput = tokens_per_second[peak_second]
	peak_active_requests = len(requests_active_per_second[peak_second])
	
	# Calculate metrics
	all_throughputs = list(tokens_per_second.values())
	avg_throughput = statistics.mean(all_throughputs)
	
	# Calculate active request metrics
	active_counts = [len(requests_active_per_second[t]) for t in tokens_per_second.keys()]
	min_active = min(active_counts) if active_counts else 0
	max_active = max(active_counts) if active_counts else 0
	avg_active = statistics.mean(active_counts) if active_counts else 0
	
	# Find ramp-up time (time to reach 80% of peak)
	sorted_times = sorted(tokens_per_second.keys())
	ramp_up_threshold = peak_throughput * 0.8
	ramp_up_time = None
	first_time = sorted_times[0] if sorted_times else 0
	
	for t in sorted_times:
		if tokens_per_second[t] >= ramp_up_threshold:
			ramp_up_time = t - first_time
			break
	
	# Calculate sustained throughput (exclude first and last 2 seconds)
	if len(sorted_times) > 4:
		sustained_times = sorted_times[2:-2]
		sustained_throughputs = [tokens_per_second[t] for t in sustained_times]
		sustained_active = [len(requests_active_per_second[t]) for t in sustained_times]
		sustained_avg = statistics.mean(sustained_throughputs) if sustained_throughputs else 0
		sustained_std = statistics.stdev(sustained_throughputs) if len(sustained_throughputs) > 1 else 0
		sustained_avg_active = statistics.mean(sustained_active) if sustained_active else 0
	else:
		sustained_avg = avg_throughput
		sustained_std = 0
		sustained_avg_active = avg_active
	
	return {
		'peak_throughput': peak_throughput,
		'peak_active_requests': peak_active_requests,
		'avg_throughput': avg_throughput,
		'sustained_avg': sustained_avg,
		'sustained_std': sustained_std,
		'sustained_avg_active': sustained_avg_active,
		'min_active': min_active,
		'max_active': max_active,
		'avg_active': avg_active,
		'ramp_up_time': ramp_up_time,
		'total_duration': max(sorted_times) - min(sorted_times) if sorted_times else 0,
		'time_series': tokens_per_second
	}

def generate_unique_filename(port: int, extension: str = "png") -> str:
    """Generate a unique filename with port and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"benchmark_port{port}_{timestamp}.{extension}"

def plot_throughput_analysis(tokens_per_second: dict, requests_active_per_second: dict, port: int, ttft_times: list = None, save_path: str = None, num_requests: int = 0, dataset_used: bool = False):
    """Plot throughput and parallel request count over time"""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return
    
    if not tokens_per_second:
        print("No data to plot.")
        return
    
    # Generate unique filename if no path provided
    if save_path is None:
        save_path = generate_unique_filename(port)
    
    # Prepare data
    sorted_times = sorted(tokens_per_second.keys())
    throughputs = [tokens_per_second[t] for t in sorted_times]
    active_counts = [len(requests_active_per_second[t]) for t in sorted_times]
    
    # Convert timestamps to relative seconds (starting from 0)
    start_time = min(sorted_times)
    relative_times = [t - start_time for t in sorted_times]
    
    # Create the plot with extra space for legend and stats
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot throughput
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Throughput (tokens/second)', color=color1)
    line1 = ax1.plot(relative_times, throughputs, color=color1, linewidth=2, label='Throughput')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for parallel requests
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Active Parallel Requests', color=color2)
    line2 = ax2.plot(relative_times, active_counts, color=color2, linewidth=2, label='Active Requests')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    workload_type = "Diverse Prompts" if dataset_used else "Single Prompt"
    plt.title(f'Throughput Analysis - Port {port} - {num_requests} Requests ({workload_type})\n{timestamp_str}', 
              fontsize=14, fontweight='bold')
    
    # Combine legends and place outside the plot area
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, 
                       bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Calculate statistics
    max_throughput = max(throughputs)
    max_active = max(active_counts)
    avg_throughput = statistics.mean(throughputs)
    avg_active = statistics.mean(active_counts)
    
    # Calculate throughput per request
    peak_second = max(tokens_per_second, key=tokens_per_second.get)
    peak_active_requests = len(requests_active_per_second[peak_second])
    throughput_per_request = max_throughput / peak_active_requests if peak_active_requests > 0 else 0
    
    # Prepare statistics text
    stats_lines = [
        "THROUGHPUT METRICS",
        "─" * 18,
        f"Peak: {max_throughput} tok/s",
        f"Avg: {avg_throughput:.1f} tok/s",
        f"Peak/Req: {throughput_per_request:.1f} tok/s/req",
        "",
        "PARALLEL REQUESTS",
        "─" * 18,
        f"Max: {max_active}",
        f"Avg: {avg_active:.1f}",
        f"Total: {num_requests}",
        "",
        "WORKLOAD",
        "─" * 18,
        f"Type: {workload_type}",
        f"Port: {port}",
    ]
    
    # Add TTFT statistics if available
    if ttft_times:
        valid_ttft = [t for t in ttft_times if t is not None]
        if valid_ttft:
            stats_lines.extend([
                "",
                "TIME TO FIRST TOKEN",
                "─" * 18,
                f"Min: {min(valid_ttft):.3f}s",
                f"Max: {max(valid_ttft):.3f}s",
                f"Median: {statistics.median(valid_ttft):.3f}s",
                f"Avg: {statistics.mean(valid_ttft):.3f}s",
                f"Success: {len(valid_ttft)}/{num_requests}"
            ])
    
    # Position statistics text further right to avoid covering axis label
    stats_text = '\n'.join(stats_lines)
    ax1.text(1.2, 0.75, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))
    
    # Adjust layout to make room for the legend and stats
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)  # Make even more room for external elements
    
    # Always save the plot with unique filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    return save_path

async def run_parallel_requests(prompts: list, single_prompt: str, client: AsyncOpenAI, num_requests: int, port: int, plot: bool = False, save_plot: str = None):
	# Shared data structures for tracking
	tokens_per_second = defaultdict(int)
	requests_active_per_second = defaultdict(set)
	
	print(f"🚀 Starting {num_requests} parallel requests...")
	if single_prompt:
		print(f"📝 Using single custom prompt: '{single_prompt[:50]}...'")
	elif prompts:
		print(f"📊 Using {len(prompts)} diverse prompts from dataset")
	else:
		print("❌ No prompts available!")
		return
	
	tasks = []
	for i in range(num_requests):
		prompt = get_prompt_for_request(prompts, i, single_prompt)
		task = asyncio.create_task(chat_integration(prompt, client, tokens_per_second, requests_active_per_second, i + 1))
		tasks.append(task)
	
	ttft_times = await asyncio.gather(*tasks)
	
	# Analyze TTFT statistics
	valid_times = [t for t in ttft_times if t is not None]
	
	if valid_times:
		print("\n" + "="*60)
		print("TIME TO FIRST TOKEN STATISTICS")
		print("="*60)
		print(f"Minimum TTFT: {min(valid_times):.3f}s")
		print(f"Maximum TTFT: {max(valid_times):.3f}s")
		print(f"Median TTFT:  {statistics.median(valid_times):.3f}s")
		print(f"Average TTFT: {statistics.mean(valid_times):.3f}s")
		print(f"Requests completed: {len(valid_times)}/{num_requests}")
	
	# Analyze throughput metrics
	throughput_metrics = analyze_throughput(tokens_per_second, requests_active_per_second)
	
	if throughput_metrics:
		print("\n" + "="*60)
		print("THROUGHPUT ANALYSIS")
		print("="*60)
		print(f"Peak Throughput:      {throughput_metrics['peak_throughput']} tokens/second")
		print(f"Requests at Peak:     {throughput_metrics['peak_active_requests']}")
		print(f"Average Throughput:   {throughput_metrics['avg_throughput']:.1f} tokens/second")
		print(f"Sustained Throughput: {throughput_metrics['sustained_avg']:.1f} ± {throughput_metrics['sustained_std']:.1f} tokens/second")
		if throughput_metrics['ramp_up_time'] is not None:
			print(f"Ramp-up Time:         {throughput_metrics['ramp_up_time']} seconds")
		print(f"Total Duration:       {throughput_metrics['total_duration']} seconds")
		
		print("\n" + "="*60)
		print("PARALLEL REQUEST ANALYSIS")
		print("="*60)
		print(f"Max Parallel Requests:      {throughput_metrics['max_active']}")
		print(f"Min Parallel Requests:      {throughput_metrics['min_active']}")
		print(f"Average Parallel Requests:  {throughput_metrics['avg_active']:.1f}")
		print(f"Sustained Avg Parallel:     {throughput_metrics['sustained_avg_active']:.1f}")
		print(f"Total Requests Started:     {num_requests}")
		
		# Calculate throughput per request metrics
		if throughput_metrics['peak_active_requests'] > 0:
			throughput_per_request = throughput_metrics['peak_throughput'] / throughput_metrics['peak_active_requests']
			print(f"Peak Throughput per Request: {throughput_per_request:.1f} tokens/second/request")
		
		# Show time series (first 15 seconds)
		print(f"\nDetailed Timeline (first 15 seconds):")
		print("  Time   | Tokens/s | Active Reqs | Tokens/s/req")
		print("  -------|----------|-------------|-------------")
		sorted_times = sorted(throughput_metrics['time_series'].keys())[:15]
		for t in sorted_times:
			throughput = throughput_metrics['time_series'][t]
			active_reqs = len(requests_active_per_second[t])
			throughput_per_req = throughput / active_reqs if active_reqs > 0 else 0
			print(f"  {t:6d} | {throughput:8d} | {active_reqs:11d} | {throughput_per_req:11.1f}")
		
		if len(throughput_metrics['time_series']) > 15:
			print(f"  ... ({len(throughput_metrics['time_series']) - 15} more seconds)")
	
	# Always generate plot for parallel requests (automatically save with unique name)
	if num_requests > 1:
		plot_path = plot_throughput_analysis(tokens_per_second, requests_active_per_second, port, ttft_times, save_plot, num_requests, bool(prompts))
	
	print("="*60)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Benchmark LLM inference performance with realistic workloads.")
	parser.add_argument("--prompt", type=str, help="Use a single custom prompt instead of diverse dataset prompts")
	parser.add_argument("--port", "-p", type=int, default=8080, help="Port number for the API server (default: 8080)")
	parser.add_argument("--num-parallel", "-n", type=int, default=1, help="Number of parallel requests to send (default: 1)")
	parser.add_argument("--dataset", type=str, default="databricks/databricks-dolly-15k", help="Hugging Face dataset to use (default: databricks/databricks-dolly-15k)")
	parser.add_argument("--max-prompts", type=int, default=100, help="Maximum number of prompts to load from dataset (default: 100)")
	parser.add_argument("--plot", action="store_true", help="Display a plot of throughput and parallel requests over time")
	parser.add_argument("--save-plot", type=str, help="Save plot to specific file (e.g., 'results.png'). If not specified, auto-generates unique filename.")
	args = parser.parse_args()

	# Load prompts before starting experiment - DEFAULT BEHAVIOR
	prompts = None
	use_single_prompt = args.prompt is not None
	
	if not use_single_prompt:
		# Default: Load diverse prompts from dataset
		print("🎯 Using diverse prompts from dataset (default behavior)")
		prompts = load_realistic_prompts(args.dataset, args.max_prompts)
		if prompts is None:
			print("⚠️  Dataset loading failed, using fallback prompt")
			use_single_prompt = True
			args.prompt = "Write a comprehensive explanation of machine learning concepts including supervised learning, unsupervised learning, and deep learning with practical examples."
	else:
		print(f"📝 Using single custom prompt: '{args.prompt[:50]}...'")

	# Create the client once
	client = AsyncOpenAI(
	    base_url=f"http://localhost:{args.port}/v1",
	)

	if args.num_parallel == 1:
		if use_single_prompt:
			prompt = args.prompt
		else:
			prompt = get_prompt_for_request(prompts, 0, args.prompt)
		asyncio.run(chat_integration(prompt, client, defaultdict(int), defaultdict(set)))
	else:
		single_prompt = args.prompt if use_single_prompt else None
		asyncio.run(run_parallel_requests(prompts, single_prompt, client, args.num_parallel, args.port, args.plot, args.save_plot))
