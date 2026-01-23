# Cross-File Reference Extraction

**Status:** ✅ Implemented

**Goal:** Improve summarization quality by providing usage context for classes, functions, and constants. This addresses poor benchmark scores in:
- `instantiation` (class): 4.3-7.6
- `collaboration` (class): 4.4-8.4
- `lifecycle` (method): 4.5-6.8
- `constraints` (constant): 3.9-8.0

## Design

### New Data Structure

```python
@dataclass
class ExtractedReference:
    """A reference to a code element from another location."""
    ref_type: str           # 'instantiation', 'method_call', 'type_hint', 'import'
    target_name: str        # 'MyClass', 'my_function'
    source_file: str        # relative path where reference occurs
    source_line: int        # line number
    context: str            # rich description: "instantiated in setup_database()"
    containing_element: str | None  # name of function/method/class containing the reference
```

### Reference Types to Extract

| Type | Python AST Node | Example | Context |
|------|-----------------|---------|---------|
| instantiation | `call` where function is identifier matching class name | `MyClass()` | "instantiated in setup_database()" |
| method_call | `call` where function is `attribute` | `obj.process()` | "method called on self.processor" |
| type_hint | `type` in parameters/return | `def foo(x: MyClass)` | "used as parameter type in foo()" |
| import | `import_from_statement` | `from x import MyClass` | "imported by file_b.py" |

### Two-Pass Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ PASS 1: Parse all files (extend existing parse_files)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  file_a.py ──parse──▶ definitions: [class MyClass, def helper]     │
│                       references:  []                               │
│                                                                     │
│  file_b.py ──parse──▶ definitions: [class Handler, def setup]      │
│                       references:  [MyClass() at line 15,          │
│                                     x: MyClass at line 20]          │
│                                                                     │
│  file_c.py ──parse──▶ definitions: [def process]                   │
│                       references:  [helper() at line 8]             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PASS 2: Link references to definitions                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Build lookup: { "MyClass": element_id_1, "helper": element_id_2 } │
│                                                                     │
│  For each reference:                                                │
│    - Find matching definition by name                               │
│    - Add to definition's context_usages with rich context          │
│                                                                     │
│  Result:                                                            │
│    MyClass.context_usages = [                                       │
│      "instantiated in setup() at file_b.py:15",                    │
│      "used as parameter type in Handler.__init__() at file_b.py:20"│
│    ]                                                                │
│    helper.context_usages = [                                        │
│      "called from process() at file_c.py:8"                        │
│    ]                                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Step 1: Add ExtractedReference dataclass

File: `src/magaldi_core/tree_sitter_manager.py`

```python
@dataclass
class ExtractedReference:
    ref_type: str           # 'instantiation', 'function_call', 'method_call', 'type_hint'
    target_name: str        # name being referenced
    line: int               # 1-indexed line number
    containing_element: str | None = None  # function/method/class name containing this ref
    context_snippet: str = ""  # brief code context
```

### Step 2: Add extract_python_references()

File: `src/magaldi_core/tree_sitter_manager.py`

Extract from AST:
1. **Call expressions** (`call` nodes):
   - Direct calls: `func()` → function_call
   - Attribute calls: `obj.method()` → method_call
   - Capitalized calls assumed class instantiation: `MyClass()` → instantiation

2. **Type hints** (`type` nodes in function parameters and return types):
   - `def foo(x: MyClass)` → type_hint for MyClass
   - `def foo() -> MyClass` → type_hint for MyClass

3. **Track containing element**: Walk up the AST to find enclosing function/method/class

### Step 3: Update ParsedFile to include references

File: `src/magaldi_core/code_parser.py`

```python
@dataclass
class ParsedFile:
    file_info: FileInfo
    elements: list[CodeElement] = field(default_factory=list)
    references: list[ExtractedReference] = field(default_factory=list)  # NEW
    parse_errors: list[str] = field(default_factory=list)
    line_count: int = 0
```

### Step 4: Add link_references() post-processing

File: `src/magaldi_core/code_parser.py`

```python
def link_references(result: ParsingResult) -> None:
    """Link extracted references to their target definitions.

    Populates context_usages on CodeElement with rich descriptions.
    """
    # Build name -> element lookup (handle duplicates by keeping all)
    definitions: dict[str, list[CodeElement]] = defaultdict(list)
    for pf in result.parsed_files:
        for elem in pf.elements:
            if elem.element_type in ('class', 'function', 'method', 'constant'):
                definitions[elem.name].append(elem)

    # Match references to definitions
    for pf in result.parsed_files:
        for ref in pf.references:
            if ref.target_name in definitions:
                # Build rich context string
                context = _build_rich_context(ref, pf.file_info.relative_path)

                # Add to all matching definitions (usually just one)
                for elem in definitions[ref.target_name]:
                    if len(elem.context_usages) < 10:  # Limit to avoid bloat
                        elem.context_usages.append(context)
```

### Step 5: Call link_references after parsing

File: `src/magaldi_core/code_parser.py`

```python
def parse_files(manifest: ChangeManifest, ...) -> ParsingResult:
    # ... existing parsing loop ...

    # NEW: Link cross-file references
    link_references(result)

    return result
```

### Step 6: Update summarization prompts to use context_usages for all types

File: `src/shared/ai/summarization.py`

Currently `context_usages` is only passed to variable/constant prompts. Extend to class/function prompts:

```python
# In build_prompt(), add for classes:
usages_section = ""
if element.context_usages and element.element_type == "class":
    usages_section = "\nWhere this class is used:\n" + "\n".join(f"- {u}" for u in element.context_usages[:5])

# Add to class template:
"class": """...
{usages_section}
..."""
```

## Rich Context Format

Examples of generated context strings:

| Reference Type | Context String |
|---------------|----------------|
| instantiation | `"instantiated in setup_database() at db/init.py:45"` |
| function_call | `"called from process_request() at handlers/api.py:123"` |
| method_call | `"method called on self.client in fetch_data() at services/http.py:67"` |
| type_hint | `"used as parameter type in create_user(user: User) at api/users.py:30"` |

## Testing

### Unit Tests

1. `test_extract_python_references()` - verify AST extraction
2. `test_link_references()` - verify matching logic
3. `test_rich_context_format()` - verify context string generation

### Integration Test

Parse a multi-file test fixture, verify:
- Class gets instantiation usages from other files
- Function gets call usages from other files
- Context strings are descriptive

## Files to Modify

1. `src/magaldi_core/tree_sitter_manager.py`
   - Add `ExtractedReference` dataclass
   - Add `extract_python_references()` function

2. `src/magaldi_core/code_parser.py`
   - Update `ParsedFile` to include references
   - Add `link_references()` function
   - Call `link_references()` in `parse_files()`

3. `src/shared/ai/summarization.py`
   - Update prompts to include `context_usages` for classes/functions

4. `tests/test_code_parser.py`
   - Add tests for reference extraction and linking

## Not in Scope (Future)

- JavaScript/TypeScript reference extraction (can add later using same pattern)
- Cross-repository references (would need ES queries)
- Inheritance tracking (`class Child(Parent)`)
- Import resolution (tracking which module `MyClass` actually comes from)
