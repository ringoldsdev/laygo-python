# Instructions for GitHub Copilot

Welcome to the laygo data processing library project! Your main goal is to help write clean, consistent, and well-tested Python code. Please adhere strictly to the following principles and conventions.

## 1. General Coding Style & Principles
* Language Version: All code should be compatible with Python 3.12+. This means you should use modern features like the | union operator for type hints and match statements where appropriate.

* Formatting: Strictly follow the PEP 8 style guide. Use a linter like ruff or flake8 to enforce this.

* Type Hinting: This is mandatory.

* Provide type hints for all function/method arguments and return values.

* Use modern type unions: int | None is preferred over Union[int, None].

* Use types from collections.abc (e.g., Iterable, Callable, Iterator) for abstract collections.

* Utilize the existing type aliases defined at the top of the file (e.g., PipelineFunction, InternalTransformer) for consistency.

* Docstrings: All public classes, methods, and functions must have Google-style docstrings.

* Include a brief one-line summary.

* Provide a more detailed explanation if necessary.

* Use Args:, Returns:, and Raises: sections to document parameters, return values, and exceptions.

Example Docstring Template:

```py
def my_method(self, parameter_a: str, parameter_b: int | None = None) -> bool:
    """A brief summary of what this method does.

    A more detailed explanation of the method's behavior, its purpose,
    and any important side effects or notes for the user.

    Args:
        parameter_a: Description of the first parameter.
        parameter_b: Description of the optional second parameter.

    Returns:
        A description of the return value, explaining what True or False means.

    Raises:
        ValueError: If parameter_a has an invalid format.
    """
    # ... implementation ...
```

* When checking code, do not worry about whitespaces. There is a formatter in place that will handle that for you.

* Don't add obvious comments. For example, avoid comments like # This is a loop or # Increment i by 1. Instead, focus on explaining why something is done, not what is done.

* Avoid using comments to disable code. If a piece of code is not needed, it should be removed entirely. Use version control to track changes instead.

* `# type: ignore` comments are ok only if there are no other options. For example, you know that the underlying code works correctly, but it's just a limitation of python in play.

## 2. Naming Conventions
* Consistency in naming is crucial for readability.

* Functions & Methods: Use snake_case (e.g., build_chunk_generator, short_circuit).

* Variables: Use snake_case (e.g., chunk_size, prev_transformer, loop_transformer).

* Classes: Use PascalCase (e.g., Transformer, ErrorHandler, PipelineContext).

* Constants: Use UPPER_SNAKE_CASE (e.g., DEFAULT_CHUNK_SIZE).

* Internal Methods/Attributes: Prefix with a single underscore _ (e.g., _pipe).

* Descriptiveness:

  * Functions used for filtering should be named predicate.

  * Functions passed to loop should be named condition.

  * Transformers passed into methods like loop or tap should be named loop_transformer or tapped_transformer.

## 3. Transformer Class Specifics
* Chainability: Every pipeline operation (map, filter, loop, etc.) must return self to allow for method chaining.

* Immutability of Logic: Operations should not modify the Transformer instance in place but rather compose a new self.transformer function by wrapping the previous one. The _pipe method is the primary mechanism for this.

* Context Awareness: When adding a new method that accepts a function (like map or filter), always check if that function is "context-aware" using the is_context_aware helper. Provide a separate execution path for both context-aware and non-aware functions.

* Overloading: For methods that can accept multiple distinct types (like tap accepting a Callable or a Transformer), use the @overload decorator to provide clear type hints for each signature.

## 4. Writing and Adding Tests

* All new functionality must be accompanied by comprehensive tests using pytest.

* File Location: Tests for laygo/transformers/transformer.py are located in tests/test_transformer.py.

* Test Organization:

  * Group related tests into classes. The class name should follow the pattern Test<FeatureGroup>, for example: TestTransformerBasics, TestTransformerOperations, TestTransformerContextSupport, TestTransformerErrorHandling

  * When adding tests for a new method, add them to the most relevant existing test class. If the method introduces a new category of functionality, create a new Test... class for it.

* Test Naming: Test methods must be descriptive and follow the pattern test_<method>_<scenario>. test_map_simple_transformation, test_loop_with_max_iterations, test_filter_with_empty_list, test_catch_with_context_aware_error

* Test Structure (Arrange-Act-Assert):

  * Arrange: Set up all necessary data, including input lists, PipelineContext objects, and the Transformer instance itself.

  * Act: Execute the transformer on the data. The result should usually be materialized into a list, e.g., result = list(transformer(data)).

  * Assert: Check that the output is correct. If the operation has side effects (like in tap or loop), assert that the side effects are also correct.

* Coverage for New Methods: When adding tests for a new method (e.g., a hypothetical my_new_op), ensure you cover:

  * Basic functionality (the "happy path").

  * Context-aware version of the functionality.

  * Edge cases, such as an empty input list ([]), a list with a single element, and a case where the operation results in an empty list.

* Interaction with other operations in a chain.

* Behavior with different chunk sizes to ensure chunking does not affect the outcome.

## 5. Documentation
* Whenever you're adding new functionality, make sure you create documentation in the wiki folder and link it in the Home.md file.

* Do not go overboard with examples. The goal is to give a clear understanding of how to use the new functionality, not to provide exhaustive examples.

## 6. Examples
* There should be an examples folder in the root of the repository.

* The folder should contain example scripts with clear names that indicate their purpose, such as example_basic_pipeline.py, example_context_aware_operations.py, and example_error_handling.py.

* Each example script should include a brief comment at the top explaining what the script demonstrates.

* The examples should be runnable as standalone scripts, meaning they should not rely on any external setup or configuration.