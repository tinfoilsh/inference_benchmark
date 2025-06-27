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

# Make heavy imports lazy - only import when needed
PLOTTING_AVAILABLE = False
DATASETS_AVAILABLE = False
TOKENIZER_AVAILABLE = False

def _lazy_import_plotting():
    """Lazy import matplotlib only when needed"""
    global PLOTTING_AVAILABLE
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        PLOTTING_AVAILABLE = True
        return plt, mdates
    except ImportError:
        PLOTTING_AVAILABLE = False
        return None, None

def _lazy_import_datasets():
    """Lazy import datasets only when needed"""
    global DATASETS_AVAILABLE
    try:
        from datasets import load_dataset
        DATASETS_AVAILABLE = True
        return load_dataset
    except ImportError:
        DATASETS_AVAILABLE = False
        return None

def _lazy_import_tokenizer():
    """Lazy import transformers only when needed"""
    global TOKENIZER_AVAILABLE
    try:
        from transformers import AutoTokenizer
        TOKENIZER_AVAILABLE = True
        return AutoTokenizer
    except ImportError:
        TOKENIZER_AVAILABLE = False
        return None

# Global tokenizer - initialize once
tokenizer = None

def initialize_tokenizer():
    """Initialize the tokenizer for input token counting"""
    global tokenizer
    AutoTokenizer = _lazy_import_tokenizer()
    if not TOKENIZER_AVAILABLE:
        print("⚠️  transformers library not available. Install with: pip install transformers")
        print("   Input token counting will be disabled.")
        return None
    
    try:
        print("🔤 Loading Qwen2.5-72B tokenizer for input token counting...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct-AWQ")
        print("✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        print("   Input token counting will be disabled.")
        return None

def count_input_tokens(prompt: str) -> int:
    """Count input tokens for a prompt"""
    if tokenizer is None:
        return 0
    
    try:
        # Format the prompt as it would be sent to the model
        messages = [{"role": "user", "content": prompt}]
        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt
        
        tokens = tokenizer.encode(formatted_prompt)
        return len(tokens)
    except Exception as e:
        print(f"Warning: Failed to count tokens for request: {e}")
        return 0

def load_realistic_prompts(dataset_name: str = "databricks/databricks-dolly-15k", max_prompts: int = 100):
    """Load diverse prompts from Hugging Face dataset"""
    load_dataset = _lazy_import_datasets()
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
	
	# Count input tokens
	input_tokens = count_input_tokens(prompt)
	if request_id is not None and input_tokens > 0:
		print(f"Input tokens: {input_tokens}")
	
	start_time = time.time()
	first_token_time = None
	previous_token_time = None
	inter_token_latencies = []
	output_tokens = 0
	
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
			current_time = time.time()
			
			if first_token_time is None:
				first_token_time = current_time
				previous_token_time = current_time
			else:
				# Calculate inter-token latency
				inter_token_latency = current_time - previous_token_time
				inter_token_latencies.append(inter_token_latency)
				previous_token_time = current_time
			
			# Count tokens and track per second (each chunk = 1 token)
			second_key = int(current_time)
			
			if chunk.choices[0].delta.content:
				output_tokens += 1
				tokens_per_second[second_key] += 1  # Each chunk = 1 token
				requests_active_per_second[second_key].add(request_id)
			
			# Only print tokens for single requests (when request_id is None)
			if request_id is None:
				print(chunk.choices[0].delta.content, end="", flush=True)
	
	if request_id is not None:
		ttft = first_token_time - start_time if first_token_time else None
		total_tokens = input_tokens + output_tokens
		if ttft:
			print(f"--- End Request {request_id} (TTFT: {ttft:.3f}s, In: {input_tokens}, Out: {output_tokens}, Total: {total_tokens}) ---\n")
		else:
			print(f"--- End Request {request_id} (No tokens received, In: {input_tokens}) ---\n")
		return ttft, inter_token_latencies, input_tokens, output_tokens
	
	# Only print newline for single requests
	if request_id is None:
		print()
	
	# Calculate TTFT for single requests too
	ttft = first_token_time - start_time if first_token_time else None
	return ttft, inter_token_latencies, input_tokens, output_tokens

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

def plot_throughput_analysis(tokens_per_second: dict, requests_active_per_second: dict, port: int, ttft_times: list = None, save_path: str = None, num_requests: int = 0, dataset_used: bool = False, inter_token_latencies: list = None, token_stats: dict = None):
    """Plot throughput and parallel request count over time"""
    plt, mdates = _lazy_import_plotting()
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return
    
    if not tokens_per_second:
        print("No data to plot.")
        return
    
    # Determine if we have a custom filename
    custom_filename_provided = save_path is not None
    
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
    
    # Add title - use filename instead of port if custom filename provided
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    workload_type = "Diverse Prompts" if dataset_used else "Single Prompt"
    
    if custom_filename_provided:
        # Extract filename without extension
        filename_base = os.path.splitext(os.path.basename(save_path))[0]
        title_identifier = filename_base
    else:
        title_identifier = f"Port {port}"
    
    plt.title(f'Throughput Analysis - {title_identifier} - {num_requests} Requests ({workload_type})\n{timestamp_str}', 
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
    peak_throughput_per_request = max_throughput / peak_active_requests if peak_active_requests > 0 else 0
    avg_throughput_per_request = avg_throughput / avg_active if avg_active > 0 else 0
    
    # Prepare statistics text
    stats_lines = [
        "THROUGHPUT METRICS",
        "─" * 18,
        f"Peak: {max_throughput} tok/s",
        f"Avg: {avg_throughput:.1f} tok/s",
        f"Peak/Req: {peak_throughput_per_request:.1f} tok/s/req",
        f"Avg/Req: {avg_throughput_per_request:.1f} tok/s/req",
        "",
        "PARALLEL REQUESTS",
        "─" * 18,
        f"Max: {max_active}",
        f"Avg: {avg_active:.1f}",
        f"Total: {num_requests}",
    ]
    
    # Add token statistics if available
    if token_stats:
        stats_lines.extend([
            "",
            "TOKEN STATISTICS",
            "─" * 18,
            f"Input: {token_stats['total_input']:,}",
            f"Output: {token_stats['total_output']:,}",
            f"Total: {token_stats['total_all']:,}",
            f"Avg In/Req: {token_stats['avg_input_per_req']:.1f}",
            f"Avg Out/Req: {token_stats['avg_output_per_req']:.1f}",
        ])
    
    stats_lines.extend([
        "",
        "WORKLOAD",
        "─" * 18,
        f"Type: {workload_type}",
        f"Port: {port}",
    ])
    
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
    
    # Add inter-token latency statistics if available
    if inter_token_latencies:
        stats_lines.extend([
            "",
            "INTER-TOKEN LATENCY",
            "─" * 18,
            f"Min: {min(inter_token_latencies)*1000:.1f}ms",
            f"Max: {max(inter_token_latencies)*1000:.1f}ms",
            f"Avg: {statistics.mean(inter_token_latencies)*1000:.1f}ms",
            f"Med: {statistics.median(inter_token_latencies)*1000:.1f}ms",
            f"P95: {sorted(inter_token_latencies)[int(0.95*len(inter_token_latencies))]*1000:.1f}ms",
            f"Measurements: {len(inter_token_latencies)}"
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
	
	results = await asyncio.gather(*tasks)
	ttft_times = [result[0] for result in results]
	all_inter_token_latencies = [result[1] for result in results]
	input_tokens_list = [result[2] for result in results]
	output_tokens_list = [result[3] for result in results]
	
	# Flatten all inter-token latencies into a single list
	all_inter_token_latencies_flat = []
	for latencies in all_inter_token_latencies:
		all_inter_token_latencies_flat.extend(latencies)
	
	# Calculate token statistics
	total_input_tokens = sum(input_tokens_list)
	total_output_tokens = sum(output_tokens_list)
	total_all_tokens = total_input_tokens + total_output_tokens
	
	valid_requests = len([t for t in ttft_times if t is not None])
	avg_input_per_req = total_input_tokens / valid_requests if valid_requests > 0 else 0
	avg_output_per_req = total_output_tokens / valid_requests if valid_requests > 0 else 0
	avg_total_per_req = total_all_tokens / valid_requests if valid_requests > 0 else 0
	
	token_stats = {
		'total_input': total_input_tokens,
		'total_output': total_output_tokens,
		'total_all': total_all_tokens,
		'avg_input_per_req': avg_input_per_req,
		'avg_output_per_req': avg_output_per_req,
		'avg_total_per_req': avg_total_per_req
	}
	
	# Display token statistics
	print("\n" + "="*60)
	print("TOKEN STATISTICS")
	print("="*60)
	print(f"Total Input Tokens:     {total_input_tokens:,}")
	print(f"Total Output Tokens:    {total_output_tokens:,}")
	print(f"Total All Tokens:       {total_all_tokens:,}")
	print(f"Average Input/Request:  {avg_input_per_req:.1f}")
	print(f"Average Output/Request: {avg_output_per_req:.1f}")
	print(f"Average Total/Request:  {avg_total_per_req:.1f}")
	if total_all_tokens > 0:
		input_ratio = (total_input_tokens / total_all_tokens) * 100
		output_ratio = (total_output_tokens / total_all_tokens) * 100
		print(f"Input/Output Ratio:     {input_ratio:.1f}% / {output_ratio:.1f}%")
	
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
	
	# Analyze inter-token latency statistics
	if all_inter_token_latencies_flat:
		print("\n" + "="*60)
		print("INTER-TOKEN LATENCY STATISTICS")
		print("="*60)
		print(f"Total inter-token measurements: {len(all_inter_token_latencies_flat)}")
		print(f"Minimum ITL: {min(all_inter_token_latencies_flat)*1000:.1f}ms")
		print(f"Maximum ITL: {max(all_inter_token_latencies_flat)*1000:.1f}ms")
		print(f"Median ITL:  {statistics.median(all_inter_token_latencies_flat)*1000:.1f}ms")
		print(f"Average ITL: {statistics.mean(all_inter_token_latencies_flat)*1000:.1f}ms")
		if len(all_inter_token_latencies_flat) > 1:
			print(f"StdDev ITL:  {statistics.stdev(all_inter_token_latencies_flat)*1000:.1f}ms")
		
		# Calculate percentiles
		sorted_latencies = sorted(all_inter_token_latencies_flat)
		p90_idx = int(0.9 * len(sorted_latencies))
		p95_idx = int(0.95 * len(sorted_latencies))
		p99_idx = int(0.99 * len(sorted_latencies))
		
		print(f"P90 ITL:     {sorted_latencies[p90_idx]*1000:.1f}ms")
		print(f"P95 ITL:     {sorted_latencies[p95_idx]*1000:.1f}ms")
		print(f"P99 ITL:     {sorted_latencies[p99_idx]*1000:.1f}ms")
		
		# Per-request statistics
		request_itl_stats = []
		for i, latencies in enumerate(all_inter_token_latencies):
			if latencies:
				request_itl_stats.append({
					'request_id': i + 1,
					'input_tokens': input_tokens_list[i],
					'output_tokens': output_tokens_list[i],
					'total_tokens': input_tokens_list[i] + output_tokens_list[i],
					'avg_itl': statistics.mean(latencies) * 1000,  # Convert to ms
					'min_itl': min(latencies) * 1000,
					'max_itl': max(latencies) * 1000
				})
		
		if request_itl_stats:
			print(f"\nPer-Request Summary (first 10 requests):")
			print("Req ID | In Tok | Out Tok| Tot Tok| Avg ITL | Min ITL | Max ITL")
			print("-------|--------|--------|--------|---------|---------|--------")
			for stats in request_itl_stats[:10]:
				print(f"  {stats['request_id']:4d} | {stats['input_tokens']:6d} | {stats['output_tokens']:6d} | {stats['total_tokens']:6d} | {stats['avg_itl']:6.1f}ms | {stats['min_itl']:6.1f}ms | {stats['max_itl']:6.1f}ms")
			
			if len(request_itl_stats) > 10:
				print(f"  ... ({len(request_itl_stats) - 10} more requests)")
	
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
			peak_throughput_per_request = throughput_metrics['peak_throughput'] / throughput_metrics['peak_active_requests']
			print(f"Peak Throughput per Request: {peak_throughput_per_request:.1f} tokens/second/request")
		
		# Calculate average throughput per request
		if throughput_metrics['avg_active'] > 0:
			avg_throughput_per_request = throughput_metrics['avg_throughput'] / throughput_metrics['avg_active']
			print(f"Avg Throughput per Request:  {avg_throughput_per_request:.1f} tokens/second/request")
		
		# Calculate sustained throughput per request
		if throughput_metrics['sustained_avg_active'] > 0:
			sustained_throughput_per_request = throughput_metrics['sustained_avg'] / throughput_metrics['sustained_avg_active']
			print(f"Sustained Throughput per Request: {sustained_throughput_per_request:.1f} tokens/second/request")
		
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
		plot_path = plot_throughput_analysis(tokens_per_second, requests_active_per_second, port, ttft_times, save_plot, num_requests, bool(prompts), all_inter_token_latencies_flat, token_stats)
	
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

	# Only initialize tokenizer for input token counting if running multiple requests
	if args.num_parallel > 1:
		initialize_tokenizer()

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
	elif args.num_parallel != 1:
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
		
		# Measure total time for throughput calculation
		start_time = time.time()
		ttft, inter_token_latencies, input_tokens, output_tokens = asyncio.run(chat_integration(prompt, client, defaultdict(int), defaultdict(set)))
		end_time = time.time()
		total_time = end_time - start_time
		
		# Display statistics for single request
		total_tokens = input_tokens + output_tokens
		
		# Display TTFT and throughput
		print("\n" + "="*60)
		print("PERFORMANCE METRICS")
		print("="*60)
		if ttft is not None:
			print(f"Time to First Token (TTFT): {ttft:.3f}s")
		else:
			print("Time to First Token (TTFT): No tokens received")
		
		print(f"Total Response Time:        {total_time:.3f}s")
		
		if output_tokens > 0 and total_time > 0:
			throughput = output_tokens / total_time
			print(f"Output Throughput:          {throughput:.1f} tokens/second")
		
		if total_tokens > 0:
			print("\n" + "="*60)
			print("TOKEN STATISTICS")
			print("="*60)
			print(f"Output tokens:  {output_tokens}")
		
		# Display inter-token latency statistics for single request
		if inter_token_latencies:
			print("\n" + "="*60)
			print("INTER-TOKEN LATENCY STATISTICS")
			print("="*60)
			print(f"Inter-token measurements: {len(inter_token_latencies)}")
			print(f"Minimum ITL: {min(inter_token_latencies)*1000:.1f}ms")
			print(f"Maximum ITL: {max(inter_token_latencies)*1000:.1f}ms")
			print(f"Median ITL:  {statistics.median(inter_token_latencies)*1000:.1f}ms")
			print(f"Average ITL: {statistics.mean(inter_token_latencies)*1000:.1f}ms")
			if len(inter_token_latencies) > 1:
				print(f"StdDev ITL:  {statistics.stdev(inter_token_latencies)*1000:.1f}ms")
	else:
		single_prompt = args.prompt if use_single_prompt else None
		asyncio.run(run_parallel_requests(prompts, single_prompt, client, args.num_parallel, args.port, args.plot, args.save_plot))
