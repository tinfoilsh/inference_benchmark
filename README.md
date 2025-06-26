# LLM Inference Benchmark Tool

A comprehensive tool for benchmarking Large Language Model (LLM) inference performance with realistic workloads. Measures throughput, latency, and parallel processing capabilities of OpenAI-compatible API servers.

## 🚀 Features

- **Realistic Workload Testing**: Uses diverse prompts from Hugging Face datasets by default
- **Comprehensive Metrics**: Time to First Token (TTFT), throughput, parallel request analysis
- **Visual Analytics**: Automatic plot generation with throughput and concurrency graphs
- **Flexible Testing**: Support for single prompts or diverse prompt sets
- **Parallel Load Testing**: Configurable concurrent request handling
- **Multiple Datasets**: Support for various HuggingFace datasets including function calling benchmarks

## 📊 What It Measures

### Core Metrics
- **Peak Throughput**: Maximum tokens per second achieved
- **Average Throughput**: Sustained token generation rate
- **Throughput per Request**: Efficiency per concurrent request
- **Time to First Token (TTFT)**: Latency metrics (min, max, median, average)
- **Parallel Request Handling**: How many requests can be processed simultaneously
- **Ramp-up Time**: Time to reach peak performance

### Visual Analytics
- Real-time throughput over time
- Active parallel requests tracking
- Performance statistics overlay
- Automatic plot saving with unique timestamps

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd inference_benchmark

# Install dependencies
pip install -r requirements.txt

# Or install individually:
pip install openai datasets matplotlib python-dotenv asyncio
```

## 📋 Requirements

- Python 3.7+
- OpenAI-compatible API server running locally
- Internet connection (for downloading datasets)

**Required packages:**
- `openai` - API client
- `datasets` - HuggingFace dataset loading
- `matplotlib` - Plot generation
- `python-dotenv` - Environment variable management

## 🎯 Usage

### Default Behavior (Recommended)
Uses diverse prompts from the Databricks Dolly dataset for realistic benchmarking:

```bash
# Basic benchmark with 10 parallel requests
python3 inference_benchmark.py -n 10

# Benchmark specific port
python3 inference_benchmark.py -p 8083 -n 20

# High load test
python3 inference_benchmark.py -n 50
```

### Custom Single Prompt
Use a specific prompt for all requests (useful for controlled testing):

```bash
# Single prompt benchmark
python3 inference_benchmark.py --prompt "Write a story about AI" -n 15

# Compare single vs diverse prompts
python3 inference_benchmark.py --prompt "What is machine learning?" -n 10
```

### Function Calling Benchmarks
Test with function calling scenarios:

```bash
# Berkeley Function Calling Leaderboard dataset
python3 inference_benchmark.py --dataset "gorilla-llm/Berkeley-Function-Calling-Leaderboard" -n 20

# API-Bank dataset
python3 inference_benchmark.py --dataset "AlibabaResearch/API-Bank" -n 15
```

### Advanced Options

```bash
# Custom dataset with specific number of prompts
python3 inference_benchmark.py --dataset "databricks/databricks-dolly-15k" --max-prompts 200 -n 30

# Save plot to specific file
python3 inference_benchmark.py -n 20 --save-plot my_benchmark.png

# Different port with custom dataset
python3 inference_benchmark.py -p 8084 --dataset "allenai/real_toxicity_prompts" -n 25
```

## 📋 Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--prompt` | | None | Use single custom prompt instead of diverse dataset |
| `--port` | `-p` | 8080 | API server port number |
| `--num-parallel` | `-n` | 1 | Number of parallel requests to send |
| `--dataset` | | `databricks/databricks-dolly-15k` | HuggingFace dataset to use |
| `--max-prompts` | | 100 | Maximum prompts to load from dataset |
| `--plot` | | False | Display plot interactively |
| `--save-plot` | | Auto-generated | Save plot to specific file |

## 🔧 Configuration

### Environment Variables
Create a `.env` file for API configuration:

```env
OPENAI_API_KEY=your_api_key_here
TINFOIL_API_KEY=your_tinfoil_key_here
```
