<!-- PROJECT_TITLE -->

# Laygo

<!-- PROJECT_TAGLINE -->

**A lightweight Python library for building resilient, in-memory data pipelines with elegant, chainable syntax.**

<!-- BADGES_SECTION -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Built with UV](https://img.shields.io/badge/built%20with-uv-green)](https://github.com/astral-sh/uv)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## 🎯 Overview

**Laygo** is a lightweight Python library for building resilient, in-memory data pipelines. It provides a fluent API to layer transformations, manage context, and handle errors with elegant, chainable syntax.

**Key Features:**

- **Fluent API**: Chainable method syntax for readable data transformations
- **Performance Optimized**: Uses chunked processing and list comprehensions for maximum speed
- **Memory Efficient**: Lazy evaluation and streaming support for large datasets
- **Parallel Processing**: Built-in ThreadPoolExecutor for CPU-intensive operations
- **Context Management**: Shared state across pipeline operations for stateful processing
- **Error Handling**: Comprehensive error handling
- **Type Safety**: Full type hints support with generic types

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
