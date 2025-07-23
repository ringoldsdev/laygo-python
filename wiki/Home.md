<!-- PROJECT_TITLE -->

# Laygo - simple pipelines, serious scale

<!-- PROJECT_TAGLINE -->

**Lightweight Python library for building resilient data pipelines with a fluent API, designed to scale effortlessly from a single script to hundreds of cores and thousands of distributed serverless functions.**

<!-- BADGES_SECTION -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Built with UV](https://img.shields.io/badge/built%20with-uv-green)](https://github.com/astral-sh/uv)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## 🎯 Overview

**Laygo** is the lightweight Python library for data pipelines that I wish existed when I first started. It's designed from the ground up to make data engineering simpler, cleaner, and more intuitive, letting you build resilient, in-memory data workflows with an elegant, fluent API.

It's built to grow with you. Scale seamlessly from a single local script to thousands of concurrent serverless functions with minimal operational overhead. Process data in parallel, branch into multiple analysis paths, and handle errors gracefully - all with the same clean, chainable syntax. 

**Key Features:**

- **Fluent & Readable**: Craft complex data transformations with a clean, chainable method syntax that's easy to write and maintain.

- **Performance Optimized**: Process data at maximum speed using chunked processing, lazy evaluation, and list comprehensions.

- **Memory Efficient**: Built-in streaming and lazy iterators allow you to handle datasets far larger than available memory.

- **Effortless Parallelism**: Accelerate CPU-intensive tasks seamlessly.

- **Fan-out Processing**: Split pipelines into multiple concurrent branches for parallel analysis of the same dataset.

- **Distributed by Design**: Your pipeline script is both the manager and the worker. When deployed as a serverless function or a container, this design allows you to scale out massively by simply running more instances of the same code. Your logic scales the same way on a thousand cores as it does on one.

- **Powerful Context Management**: Share state and configuration across your entire pipeline for advanced, stateful processing.

- **Resilient Error Handling**: Isolate and manage errors at the chunk level, preventing a single bad record from failing your entire job.

- **Modern & Type-Safe**: Leverage full support for modern Python with generic type hints for robust, maintainable code.

---

## 📦 Installation

```bash
pip install laygo
```

Or for development:

```bash
git clone https://github.com/ringoldsdev/laygo-python.git
cd laygo-python
pip install -e ".[dev]"
```

### 🐳 Dev Container Setup

If you're using this project in a dev container, you'll need to configure Git to use HTTPS instead of SSH for authentication:

```bash
# Switch to HTTPS remote URL
git remote set-url origin https://github.com/ringoldsdev/laygo-python.git

# Configure Git to use HTTPS for all GitHub operations
git config --global url."https://github.com/".insteadOf "git@github.com:"
```

---

## ▶️ Usage

### Basic Pipeline Operations

```python
from laygo import Pipeline

# Simple data transformation
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = (
    Pipeline(data)
    .transform(lambda t: t.filter(lambda x: x % 2 == 0))  # Keep even numbers
    .transform(lambda t: t.map(lambda x: x * 2))          # Double them
    .to_list()
)
print(result)  # [4, 8, 12, 16, 20]
```

### Context-Aware Operations

```python
from laygo import Pipeline
from laygo import PipelineContext

# Create context with shared state
context: PipelineContext = {"multiplier": 3, "threshold": 10}

result = (
    Pipeline([1, 2, 3, 4, 5])
    .context(context)
    .transform(lambda t: t.map(lambda x, ctx: x * ctx["multiplier"]))
    .transform(lambda t: t.filter(lambda x, ctx: x > ctx["threshold"]))
    .to_list()
)
print(result)  # [12, 15]
```

### ETL Pipeline Example

```python
from laygo import Pipeline

# Sample employee data processing
employees = [
    {"name": "Alice", "age": 25, "salary": 50000},
    {"name": "Bob", "age": 30, "salary": 60000},
    {"name": "Charlie", "age": 35, "salary": 70000},
    {"name": "David", "age": 28, "salary": 55000},
]

# Extract, Transform, Load pattern
high_earners = (
    Pipeline(employees)
    .transform(lambda t: t.filter(lambda emp: emp["age"] > 28))           # Extract
    .transform(lambda t: t.map(lambda emp: {                             # Transform
        "name": emp["name"],
        "annual_salary": emp["salary"],
        "monthly_salary": emp["salary"] / 12
    }))
    .transform(lambda t: t.filter(lambda emp: emp["annual_salary"] > 55000)) # Filter
    .to_list()
)
```

### Using Transformers Directly

```python
from laygo import Transformer

# Create a reusable transformation pipeline
transformer = (
    Transformer.init(int)
    .filter(lambda x: x % 2 == 0)   # Keep even numbers
    .map(lambda x: x * 2)           # Double them
    .filter(lambda x: x > 5)        # Keep > 5
)

# Apply to different datasets
result1 = list(transformer([1, 2, 3, 4, 5]))  # [4, 8]
result2 = list(transformer(range(10)))          # [4, 8, 12, 16, 20]
```

### Custom Transformer Composition

```python
from laygo import Pipeline
from laygo import Transformer

# Create reusable transformation components
validate_data = Transformer.init(dict).filter(lambda x: x.get("id") is not None)
normalize_text = Transformer.init(dict).map(lambda x: {**x, "name": x["name"].strip().title()})

# Use transformers directly with Pipeline.transform()
result = (
    Pipeline(raw_data)
    .transform(validate_data)      # Pass transformer directly
    .transform(normalize_text)     # Pass transformer directly
    .to_list()
)
```

### Parallel Processing

```python
from laygo import Pipeline
from laygo import ParallelTransformer

# Process large datasets with multiple threads
large_data = range(100_000)

# Create parallel transformer
parallel_processor = (
  ParallelTransformer.init(
    int,
    max_workers=4,
    ordered=True,    # Maintain result order
    chunk_size=10000 # Process in chunks
  ).map(lambda x: x ** 2)
)

results = (
    Pipeline(large_data)
    .transform(parallel_processor)
    .transform(lambda t: t.filter(lambda x: x > 100))
    .first(1000)  # Get first 1000 results
)
```

### Pipeline Branching (Fan-out Processing)

```python
from laygo import Pipeline
from laygo.transformers.transformer import createTransformer

# Sample data: customer orders
orders = [
    {"id": 1, "customer": "Alice", "amount": 150, "product": "laptop"},
    {"id": 2, "customer": "Bob", "amount": 25, "product": "book"},
    {"id": 3, "customer": "Charlie", "amount": 75, "product": "headphones"},
    {"id": 4, "customer": "Diana", "amount": 200, "product": "monitor"},
    {"id": 5, "customer": "Eve", "amount": 30, "product": "mouse"},
]

# Create different analysis branches
high_value_analysis = (
    createTransformer(dict)
    .filter(lambda order: order["amount"] > 100)
    .map(lambda order: {
        "customer": order["customer"],
        "amount": order["amount"],
        "category": "high_value"
    })
)

product_summary = (
    createTransformer(dict)
    .map(lambda order: {"product": order["product"], "count": 1})
    # Group by product and sum counts (simplified example)
)

customer_spending = (
    createTransformer(dict)
    .map(lambda order: {
        "customer": order["customer"],
        "total_spent": order["amount"]
    })
)

# Branch the pipeline into multiple concurrent analyses
results = Pipeline(orders).branch({
    "high_value_orders": high_value_analysis,
    "products": product_summary,
    "customer_totals": customer_spending
})

print("High value orders:", results["high_value_orders"])
# [{'customer': 'Alice', 'amount': 150, 'category': 'high_value'}, 
#  {'customer': 'Diana', 'amount': 200, 'category': 'high_value'}]

print("Product analysis:", len(results["products"]))
# 5 (all products processed)

print("Customer spending:", len(results["customer_totals"]))  
# 5 (all customers processed)
```

### Advanced Branching with Error Isolation

```python
from laygo import Pipeline
from laygo.transformers.transformer import createTransformer

# Data with potential issues
mixed_data = [1, 2, "invalid", 4, 5, None, 7, 8]

# Branch 1: Safe numeric processing
safe_numbers = (
    createTransformer(int | str | None)
    .filter(lambda x: isinstance(x, int) and x is not None)
    .map(lambda x: x * 2)
)

# Branch 2: String processing with error handling
string_processing = (
    createTransformer(int | str | None)
    .filter(lambda x: isinstance(x, str))
    .map(lambda x: f"processed_{x}")
    .catch(lambda t: t.map(lambda x: "error_handled"))
)

# Branch 3: Statistical analysis
stats_analysis = (
    createTransformer(int | str | None)
    .filter(lambda x: isinstance(x, int) and x is not None)
    .map(lambda x: x)  # Pass through for stats
)

# Execute all branches concurrently
results = Pipeline(mixed_data).branch({
    "numbers": safe_numbers,
    "strings": string_processing,
    "stats": stats_analysis
}, batch_size=100)

print("Processed numbers:", results["numbers"])  # [2, 4, 8, 10, 14, 16]
print("Processed strings:", results["strings"])  # ['processed_invalid']
print("Stats data:", results["stats"])           # [1, 2, 4, 5, 7, 8]

# Each branch processes the complete dataset independently
# Errors in one branch don't affect others
```

### Error Handling and Recovery

```python
from laygo import Pipeline
from laygo import Transformer

def risky_operation(x):
    if x == 5:
        raise ValueError("Cannot process 5")
    return x * 2

def error_handler(chunk, error, context):
    print(f"Error in chunk {chunk}: {error}")
    return [0] * len(chunk)  # Return default values

# Pipeline with error recovery
result = (
    Pipeline([1, 2, 3, 4, 5, 6])
    .transform(lambda t: t.map(risky_operation).catch(
        lambda sub_t: sub_t.map(lambda x: x + 1),
        on_error=error_handler
    ))
    .to_list()
)
```

---

## ⚙️ Projects using Laygo

- **[Efemel](https://github.com/ringoldsdev/efemel)** - A CLI tool that processes Python files as configuration markup and exports them to JSON/YAML, replacing traditional templating DSLs with native Python syntax.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🚀 Built With

- **[Python 3.12+](https://python.org)** - Core language with modern type hints
- **[Ruff](https://github.com/astral-sh/ruff)** - Code formatting and linting
- **[Pytest](https://pytest.org/)** - Testing framework
- **[DevContainers](https://containers.dev/)** - Consistent development environment
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD automation

---

**⭐ Star this repository if Laygo helps your data processing workflows!**
