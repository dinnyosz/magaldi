# Variable/Constant Usefulness Filter

## Status: ✅ Implemented (All Languages)

## Problem

We extract many variables and constants during parsing, but most are transient/temporary values that provide little value for code discovery:
- Instance creations: `client = SomeClient()`
- Function call results: `data = process_data()`
- Temporary variables: `result = fetch_items()`

These clutter the index and waste storage/tokens without adding searchable value.

## Solution

Add a usefulness filter at parse time that skips variables/constants unlikely to be useful for code discovery. The filter is implemented as a shared module (`usefulness_filter.py`) used by all language extractors.

## Language Analysis

| Language | Implementation Status |
|----------|----------------------|
| **Python** | ✅ Filter implemented - `_is_useful_assignment()` |
| **JavaScript** | ✅ Filter implemented - `_extract_js_variable()` |
| **PHP** | ✅ Filter implemented - `_extract_php_variable()` |
| **Rust** | ✅ Filter implemented - `_extract_rust_variable()`, `_extract_rust_module_const()`, `_extract_rust_static()` |

## Usefulness Criteria

### KEEP (Useful)

| Category | Examples | Detection |
|----------|----------|-----------|
| **Constants by convention** | `MAX_RETRIES`, `API_URL` | `name.isupper() and len(name) > 1` |
| **Literal values** | `"active"`, `42`, `[1,2,3]`, `{"key": "val"}` | `value_type in (string, integer, list, dict, ...)` |
| **Type aliases** | `UserID = int`, `Callback = Callable[...]` | `value_type in (identifier, subscript)` and no call |
| **Enum definitions** | `Status = Enum("Status", ...)` | Call to `Enum`, `IntEnum`, `StrEnum`, `Flag` |
| **TypeVar definitions** | `T = TypeVar("T")` | Call to `TypeVar`, `ParamSpec`, `TypeVarTuple` |
| **Named tuples** | `Point = namedtuple("Point", ...)` | Call to `namedtuple`, `NamedTuple` |
| **Compiled patterns** | `PATTERN = re.compile(...)` | Call to `re.compile`, `compile` |
| **Loggers** | `logger = logging.getLogger(...)` | Call to `getLogger`, `logging.getLogger` |
| **Threading primitives** | `_lock = threading.Lock()` | Call to `Lock`, `RLock`, `Semaphore`, etc. |
| **Singletons/caches** | `_instance = None`, `_cache = {}` | Literal None or empty dict/list |
| **Arrow functions (JS/TS)** | `const add = (a, b) => a + b` | `value_type == "arrow_function"` |
| **Regex literals (JS)** | `const PATTERN = /regex/` | `value_type == "regex"` |
| **as const (TS)** | `const DIR = {...} as const` | `value_type == "as_expression"` |

### SKIP (Not Useful)

| Category | Examples | Detection |
|----------|----------|-----------|
| **Instance creation** | `client = HttpClient()` | Call to PascalCase name |
| **Factory method calls** | `service = Factory.create()` | Attribute call (has `.`) |
| **Function results** | `data = process_items()` | Call to lowercase function |
| **Method call results** | `response = requests.get(url)` | Attribute call |
| **Constructor calls (JS)** | `new SomeClass()` | `new_expression` (except Map, Set, RegExp) |
| **Short temp names** | `i`, `j`, `x`, `tmp` | Already filtered |

## Implementation

### ✅ Phase 1: Python Extractor (DONE)

Modified `_extract_python_assignment()` in `src/magaldi_core/extractors/python.py`.

**Key additions:**
- `USEFUL_FACTORIES` - set of factory function names that produce useful values (Enum, TypeVar, namedtuple, etc.)
- `USEFUL_ATTRIBUTE_FACTORIES` - set of attribute-style factories (re.compile, logging.getLogger, etc.)
- `SKIP_NAMES` - set of short/temp names to always skip (i, j, tmp, temp, etc.)
- `_is_useful_assignment()` - determines if an assignment is useful for code discovery
- `ExtractionStats` / `SkippedVariable` - data classes for tracking what was skipped
- Logging at DEBUG level shows skipped variables after each file

**Example output:**
```
Variable filter for test.py: kept=8, skipped=4
  SKIP: client (line 13) - instance_creation [call] HttpClient()
  SKIP: service (line 14) - method_call_result [call] ServiceFactory.create()
  SKIP: data (line 15) - function_call_result [call] process_items()
  SKIP: response (line 16) - method_call_result [call] requests.get(url)
```

**Original code reference:**

```python
def _extract_python_assignment(
    node: Node, lines: list[str], is_module_level: bool = False, parent_class: Node | None = None
) -> ExtractedElement | None:
    """Extract a variable/constant assignment."""
    left_node = get_child_by_field(node, "left")
    if not left_node or left_node.type != "identifier":
        return None

    name = get_node_text(left_node)

    # Skip common non-interesting patterns
    if name in ("i", "j", "k", "x", "y", "z", "_", "self", "cls"):
        return None

    # Get the right-hand side value
    right_node = get_child_by_field(node, "right")

    # Apply usefulness filter
    if not _is_useful_assignment(name, right_node, is_module_level):
        return None

    # ... rest of extraction
```

Add new helper function:

```python
# Useful factory function patterns
USEFUL_FACTORIES = frozenset({
    # Enums
    "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
    # Typing
    "TypeVar", "ParamSpec", "TypeVarTuple", "NewType",
    # Collections
    "namedtuple", "NamedTuple", "defaultdict", "OrderedDict",
    # Regex
    "compile",  # re.compile
    # Logging
    "getLogger",
    # Threading
    "Lock", "RLock", "Semaphore", "BoundedSemaphore", "Condition", "Event",
    # Paths
    "Path", "PurePath",
    # Dataclass
    "field",
})

USEFUL_ATTRIBUTE_FACTORIES = frozenset({
    "re.compile",
    "logging.getLogger",
    "threading.Lock",
    "threading.RLock",
    "pathlib.Path",
})


def _is_useful_assignment(name: str, value_node: Node | None, is_module_level: bool) -> bool:
    """Determine if a variable assignment is useful for code discovery."""
    if not value_node:
        return False

    # USEFUL: Constants by naming convention (UPPER_CASE)
    if name.isupper() and len(name) > 1:
        return True

    # USEFUL: Literal values (configuration, data)
    literal_types = {"string", "integer", "float", "true", "false", "none",
                     "list", "dictionary", "tuple", "set", "concatenated_string"}
    if value_node.type in literal_types:
        return True

    # USEFUL: Type references (type aliases)
    if value_node.type in ("identifier", "subscript", "attribute"):
        # Type alias like: UserID = int, OptionalStr = Optional[str]
        # These don't have function calls
        return True

    # Check call expressions
    if value_node.type == "call":
        func_node = get_child_by_field(value_node, "function")
        if func_node:
            func_text = get_node_text(func_node)

            # USEFUL: Known useful factory functions
            if func_text in USEFUL_FACTORIES:
                return True

            # USEFUL: Known useful attribute factories
            if func_text in USEFUL_ATTRIBUTE_FACTORIES:
                return True

            # SKIP: Instance creation (PascalCase class name)
            if func_node.type == "identifier":
                # SomeClient(), Handler(), etc.
                if func_text and func_text[0].isupper() and not func_text.isupper():
                    return False

            # SKIP: Method calls on objects (attribute)
            if func_node.type == "attribute":
                # requests.get(), json.loads(), factory.create()
                return False

            # SKIP: Function call results (lowercase function)
            if func_node.type == "identifier" and func_text and func_text[0].islower():
                return False

    # Default: keep if module-level, skip if local
    return is_module_level
```

### ✅ Phase 2: JavaScript/TypeScript Extractor (DONE)

Modified `extract_javascript_elements()` in `src/magaldi_core/extractors/javascript.py`.

**Key additions:**
- Now extracts all `const`/`let`/`var` declarations with usefulness filter
- `_extract_js_variable()` - extracts variables, applies usefulness filter
- Handles `new_expression` by looking at "constructor" field for class name

### ✅ Phase 3: PHP Extractor (DONE)

Modified `extract_php_elements()` in `src/magaldi_core/extractors/php.py`.

**Key additions:**
- Now extracts all assignment expressions with usefulness filter
- `_extract_php_variable()` - handles PHP-specific AST nodes:
  - `object_creation_expression` (new SomeClass())
  - `function_call_expression` (someFunction())
  - `member_call_expression` ($obj->method())
  - `scoped_call_expression` (SomeClass::staticMethod())

### ✅ Phase 4: Rust Extractor (DONE)

Modified `extract_rust_elements()` in `src/magaldi_core/extractors/rust.py`.

**Key additions:**
- Now extracts module-level `const_item`, `static_item`, and `let_declaration`
- `_extract_rust_module_const()` - extracts const with usefulness filter
- `_extract_rust_static()` - extracts static with usefulness filter
- `_extract_rust_variable()` - extracts let declarations with usefulness filter

## Testing

Add test cases in `tests/extractors/test_variable_usefulness.py`:

```python
class TestPythonVariableUsefulness:
    def test_keeps_constants(self):
        """UPPER_CASE names are always kept."""

    def test_keeps_literals(self):
        """Literal values (strings, numbers, lists, dicts) are kept."""

    def test_keeps_type_aliases(self):
        """Type aliases like UserID = int are kept."""

    def test_keeps_enum_definitions(self):
        """Enum() calls are kept."""

    def test_keeps_typevar(self):
        """TypeVar() definitions are kept."""

    def test_skips_instance_creation(self):
        """new SomeClass() style calls are skipped."""

    def test_skips_method_calls(self):
        """obj.method() results are skipped."""

    def test_skips_function_results(self):
        """lowercase_func() results are skipped."""
```

## Migration

No migration needed - this only affects new parses. Existing indexed data remains unchanged.

## Metrics

Track before/after:
- Total variable/constant count per repo
- Storage size reduction
- Expected: 50-70% reduction in variable elements

## Edge Cases

1. **Factory functions that return useful objects**: `settings = Settings()` - skip (instance)
2. **Builder patterns**: `config = ConfigBuilder().with_x().build()` - skip (method chain)
3. **Module-level None/empty**: `_cache = {}` - keep (singleton pattern)
4. **Dunder names**: `__all__ = [...]` - keep (module metadata)
5. **Private with literals**: `_DEFAULT = "value"` - keep (has literal value)
