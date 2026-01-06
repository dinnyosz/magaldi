# Magaldi Parser - Phase 3: Parsing

## Overview

The Parsing phase uses Tree-sitter to extract structured code elements from files identified in the Change Manifest. It builds parent-child relationships and prepares elements for storage.

```
┌─────────────────────────────────────────────────────────────────┐
│                       PHASE 3: PARSING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  3.1              3.2              3.3              3.4          │
│  ─────            ─────            ─────            ─────        │
│  INITIALIZE   →   PARSE        →   EXTRACT      →   BUILD       │
│  PARSERS          FILES            ELEMENTS         HIERARCHY    │
│                                                                 │
│  • Load grammars  • Read content   • Classes        • Parent IDs │
│  • Load queries   • Create AST     • Functions      • Levels     │
│  • Per language   • Walk tree      • Methods        • Element IDs│
│                   • Apply queries  • Variables      • Signatures │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Input

From Phase 2 (Change Detection):

```python
@dataclass
class ChangeManifest:
    scope: str
    repository: str
    username: str

    new_files: List[FileInfo]
    modified_files: List[FileInfo]
    deleted_files: List[FileInfo]

@dataclass
class FileInfo:
    relative_path: str
    absolute_path: str
    language: str
    hash: str
```

---

## 3.1 Initialize Parsers

### Purpose

Load Tree-sitter grammars and query files for each language present in the manifest.

### Language Registry

```python
LANGUAGE_CONFIG = {
    'python': {
        'grammar': 'tree_sitter_python',
        'query_file': 'queries/python.scm',
        'extensions': ['.py'],
    },
    'javascript': {
        'grammar': 'tree_sitter_javascript',
        'query_file': 'queries/javascript.scm',
        'extensions': ['.js', '.jsx', '.mjs', '.cjs'],
    },
    'typescript': {
        'grammar': 'tree_sitter_typescript',
        'query_file': 'queries/typescript.scm',
        'extensions': ['.ts', '.tsx'],
    },
    'php': {
        'grammar': 'tree_sitter_php',
        'query_file': 'queries/php.scm',
        'extensions': ['.php'],
    },
    'rust': {
        'grammar': 'tree_sitter_rust',
        'query_file': 'queries/rust.scm',
        'extensions': ['.rs'],
    },
}
```

### Parser Pool

```python
class ParserPool:
    """Manages Tree-sitter parsers for multiple languages"""

    def __init__(self):
        self.parsers: Dict[str, tree_sitter.Parser] = {}
        self.queries: Dict[str, tree_sitter.Query] = {}

    def get_parser(self, language: str) -> tree_sitter.Parser:
        """Get or create parser for language"""
        if language not in self.parsers:
            self._load_language(language)
        return self.parsers[language]

    def get_query(self, language: str) -> tree_sitter.Query:
        """Get query for language"""
        if language not in self.queries:
            self._load_language(language)
        return self.queries[language]

    def _load_language(self, language: str):
        """Load grammar and query for language"""
        config = LANGUAGE_CONFIG[language]

        # Load grammar
        lang_module = importlib.import_module(config['grammar'])
        lang = lang_module.language()

        parser = tree_sitter.Parser()
        parser.language = lang
        self.parsers[language] = parser

        # Load query
        query_path = Path(config['query_file'])
        query_text = query_path.read_text()
        self.queries[language] = lang.query(query_text)
```

### Lazy Loading

- Only load parsers for languages present in manifest
- Cache parsers for reuse across files
- Thread-safe for parallel processing

---

## 3.2 Parse Files

### Purpose

Read file content and create Abstract Syntax Tree (AST) using Tree-sitter.

### Process

```python
def parse_file(file_info: FileInfo, parser_pool: ParserPool) -> ParseResult:
    """Parse a single file into AST"""

    # 1. Read file content
    try:
        content = Path(file_info.absolute_path).read_bytes()
    except IOError as e:
        return ParseResult(error=f"Cannot read file: {e}")

    # 2. Get parser for language
    parser = parser_pool.get_parser(file_info.language)

    # 3. Parse into AST
    tree = parser.parse(content)

    # 4. Check for errors
    if tree.root_node.has_error:
        # Log warning but continue - Tree-sitter handles partial parses
        log.warning(f"Syntax errors in {file_info.relative_path}")

    return ParseResult(
        tree=tree,
        content=content,
        file_info=file_info
    )
```

### Error Handling

| Scenario | Action |
|----------|--------|
| File unreadable | Log error, skip file, continue |
| Syntax errors | Log warning, parse anyway (Tree-sitter is fault-tolerant) |
| Binary file | Detect via null bytes, skip |
| Empty file | Return empty element list |
| Encoding issues | Try UTF-8, fallback to latin-1, then skip |

### Binary Detection

```python
def is_binary(content: bytes, sample_size: int = 8192) -> bool:
    """Detect binary files by checking for null bytes"""
    sample = content[:sample_size]
    return b'\x00' in sample
```

---

## 3.3 Extract Elements

### Purpose

Use Tree-sitter queries to extract structured code elements from AST.

### Element Types

| Type | Level | Description |
|------|-------|-------------|
| `file` | 0 | The file itself (virtual element) |
| `class` | 1 | Class definitions |
| `function` | 2 | Standalone functions |
| `method` | 2 | Class methods |
| `variable` | 3 | Module-level variables, constants |

### Query Structure (Python Example)

```scheme
; queries/python.scm

; Classes
(class_definition
  name: (identifier) @class.name
  body: (block) @class.body
) @class.definition

; Functions (standalone)
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  return_type: (type)? @function.return_type
  body: (block) @function.body
) @function.definition

; Methods (inside class)
(class_definition
  body: (block
    (function_definition
      name: (identifier) @method.name
      parameters: (parameters) @method.params
      return_type: (type)? @method.return_type
      body: (block) @method.body
    ) @method.definition
  )
)

; Decorators
(decorated_definition
  (decorator
    (identifier) @decorator.name
  )?
  (decorator
    (call
      function: (identifier) @decorator.call_name
    )
  )?
  definition: (_) @decorator.target
) @decorated

; Docstrings
(expression_statement
  (string) @docstring
) @docstring.statement

; Module-level assignments (variables/constants)
(module
  (expression_statement
    (assignment
      left: (identifier) @variable.name
      right: (_) @variable.value
    )
  ) @variable.definition
)

; Imports
(import_statement) @import
(import_from_statement) @import.from

; Type hints
(type) @type_hint
```

### Extraction Process

```python
def extract_elements(parse_result: ParseResult, parser_pool: ParserPool) -> List[RawElement]:
    """Extract elements from parsed AST"""

    query = parser_pool.get_query(parse_result.file_info.language)
    captures = query.captures(parse_result.tree.root_node)

    elements = []

    # Group captures by element
    element_groups = group_captures(captures)

    for group in element_groups:
        element = build_element(group, parse_result)
        if element:
            elements.append(element)

    return elements


def build_element(captures: Dict[str, Node], parse_result: ParseResult) -> RawElement:
    """Build element from captured nodes"""

    content = parse_result.content

    # Determine element type from captures
    if 'class.definition' in captures:
        return build_class_element(captures, content, parse_result.file_info)
    elif 'method.definition' in captures:
        return build_method_element(captures, content, parse_result.file_info)
    elif 'function.definition' in captures:
        return build_function_element(captures, content, parse_result.file_info)
    elif 'variable.definition' in captures:
        return build_variable_element(captures, content, parse_result.file_info)

    return None
```

### Element Data Structure

```python
@dataclass
class RawElement:
    # Identity (set in Phase 3.4)
    element_id: Optional[str] = None

    # Location
    scope: str
    repository: str
    username: str
    relative_path: str

    # Element info
    element_type: str          # 'class', 'function', 'method', 'variable'
    name: str
    language: str

    # Position
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int

    # Content
    raw_code: str              # Full source code of element
    signature: Optional[str]   # Function/method signature
    docstring: Optional[str]   # Extracted docstring

    # Metadata
    decorators: List[str]      # ['staticmethod', 'property', ...]
    is_async: bool
    visibility: str            # 'public', 'private', 'protected'

    # Hierarchy (set in Phase 3.4)
    level: int = 0
    parent_id: Optional[str] = None

    # Type info (language-specific)
    return_type: Optional[str]
    parameters: List[Dict]     # [{'name': 'x', 'type': 'int', 'default': None}, ...]
```

### Language-Specific Extractors

Each language has specific extraction logic:

#### Python

```python
def build_function_element(captures, content, file_info) -> RawElement:
    """Build function element for Python"""

    func_node = captures['function.definition']
    name_node = captures['function.name']
    params_node = captures.get('function.params')
    return_node = captures.get('function.return_type')

    # Extract code
    raw_code = content[func_node.start_byte:func_node.end_byte].decode('utf-8')

    # Build signature
    signature = f"def {name_node.text.decode('utf-8')}"
    if params_node:
        signature += params_node.text.decode('utf-8')
    if return_node:
        signature += f" -> {return_node.text.decode('utf-8')}"

    # Find docstring (first string in body)
    docstring = extract_docstring(func_node, content)

    # Check for async
    is_async = func_node.parent and func_node.parent.type == 'async_function_definition'

    # Extract decorators
    decorators = extract_decorators(func_node, content)

    # Determine visibility
    name = name_node.text.decode('utf-8')
    visibility = 'private' if name.startswith('_') else 'public'

    return RawElement(
        scope=file_info.scope,
        repository=file_info.repository,
        username=file_info.username,
        relative_path=file_info.relative_path,
        element_type='function',
        name=name,
        language='python',
        line_start=func_node.start_point[0] + 1,
        line_end=func_node.end_point[0] + 1,
        byte_start=func_node.start_byte,
        byte_end=func_node.end_byte,
        raw_code=raw_code,
        signature=signature,
        docstring=docstring,
        decorators=decorators,
        is_async=is_async,
        visibility=visibility,
        return_type=return_node.text.decode('utf-8') if return_node else None,
        parameters=extract_parameters(params_node, content),
    )
```

#### JavaScript/TypeScript

```python
def build_function_element_js(captures, content, file_info) -> RawElement:
    """Build function element for JavaScript/TypeScript"""

    # Handle multiple function forms:
    # - function declarations: function foo() {}
    # - arrow functions: const foo = () => {}
    # - method definitions: { foo() {} }
    # - class methods: class X { foo() {} }

    # ... similar structure to Python
```

### Docstring Extraction

```python
def extract_docstring(element_node: Node, content: bytes) -> Optional[str]:
    """Extract docstring from element"""

    # Python: First expression statement that's a string
    body = find_child(element_node, 'block')
    if body and body.children:
        first_stmt = body.children[0]
        if first_stmt.type == 'expression_statement':
            string_node = find_child(first_stmt, 'string')
            if string_node:
                raw = string_node.text.decode('utf-8')
                # Strip quotes
                return raw.strip('"""').strip("'''").strip('"').strip("'").strip()

    return None
```

### Decorator Extraction

```python
def extract_decorators(element_node: Node, content: bytes) -> List[str]:
    """Extract decorator names"""

    decorators = []

    # Check if parent is decorated_definition
    if element_node.parent and element_node.parent.type == 'decorated_definition':
        for child in element_node.parent.children:
            if child.type == 'decorator':
                # Get decorator name
                name = extract_decorator_name(child, content)
                if name:
                    decorators.append(name)

    return decorators
```

---

## 3.4 Build Hierarchy

### Purpose

Establish parent-child relationships and generate unique element IDs.

### Hierarchy Levels

```
Level 0: file
    │
    ├── Level 1: class
    │       │
    │       └── Level 2: method
    │
    ├── Level 2: function (standalone)
    │
    └── Level 3: variable (module-level)
```

### Element ID Format

```
{scope}:{repository}:{username}:{relative_path}:{type}:{name}:{line}

Examples:
backend:auth-service:main:src/auth/login.py:file:login.py:1
backend:auth-service:main:src/auth/login.py:class:AuthService:15
backend:auth-service:main:src/auth/login.py:method:authenticate:25
backend:auth-service:main:src/auth/login.py:function:hash_password:100
backend:auth-service:alice:src/auth/login.py:method:authenticate:25
```

### Hierarchy Building

```python
def build_hierarchy(elements: List[RawElement], file_info: FileInfo) -> List[RawElement]:
    """Assign parent IDs and levels to elements"""

    # Create file-level element
    file_element = RawElement(
        scope=file_info.scope,
        repository=file_info.repository,
        username=file_info.username,
        relative_path=file_info.relative_path,
        element_type='file',
        name=Path(file_info.relative_path).name,
        language=file_info.language,
        line_start=1,
        line_end=count_lines(file_info.absolute_path),
        level=0,
        parent_id=None,
    )
    file_element.element_id = generate_element_id(file_element)

    # Sort elements by position
    elements.sort(key=lambda e: (e.line_start, e.line_end))

    # Build containment tree
    result = [file_element]
    class_map = {}  # line_start -> class element

    for element in elements:
        # Assign level
        if element.element_type == 'class':
            element.level = 1
            element.parent_id = file_element.element_id
            class_map[element.line_start] = element

        elif element.element_type == 'method':
            element.level = 2
            # Find containing class
            parent_class = find_containing_class(element, class_map)
            element.parent_id = parent_class.element_id if parent_class else file_element.element_id

        elif element.element_type == 'function':
            element.level = 2
            element.parent_id = file_element.element_id

        elif element.element_type == 'variable':
            element.level = 3
            element.parent_id = file_element.element_id

        # Generate element ID
        element.element_id = generate_element_id(element)
        result.append(element)

    return result


def find_containing_class(element: RawElement, class_map: Dict) -> Optional[RawElement]:
    """Find the class that contains this element"""

    for class_start, class_element in class_map.items():
        if (class_element.line_start <= element.line_start and
            class_element.line_end >= element.line_end):
            return class_element

    return None


def generate_element_id(element: RawElement) -> str:
    """Generate unique element ID"""

    return ":".join([
        element.scope,
        element.repository,
        element.username,
        element.relative_path,
        element.element_type,
        element.name,
        str(element.line_start)
    ])
```

### Handling Nested Classes

```python
# For languages with nested classes (Python, JS)
def build_hierarchy_nested(elements: List[RawElement]) -> List[RawElement]:
    """Handle arbitrarily nested class structures"""

    # Sort by (start, -end) to process outer elements first
    elements.sort(key=lambda e: (e.line_start, -e.line_end))

    # Stack-based containment
    stack = []  # [(element, end_line), ...]

    for element in elements:
        # Pop elements that don't contain this one
        while stack and stack[-1][1] < element.line_start:
            stack.pop()

        # Parent is top of stack (or file)
        if stack:
            element.parent_id = stack[-1][0].element_id
            element.level = stack[-1][0].level + 1
        else:
            element.level = 1  # Direct child of file

        # Push classes onto stack
        if element.element_type == 'class':
            stack.append((element, element.line_end))
```

---

## Output

Phase 3 produces a list of fully-populated `RawElement` objects:

```python
@dataclass
class ParsedFile:
    file_info: FileInfo
    file_hash: str
    elements: List[RawElement]
    parse_errors: List[str]
    line_count: int

@dataclass
class ParsingResult:
    scope: str
    repository: str
    username: str

    parsed_files: List[ParsedFile]
    failed_files: List[FailedFile]

    total_elements: int
    elements_by_type: Dict[str, int]
    elements_by_language: Dict[str, int]
```

This is passed to Phase 4: Storage.

---

## Progress Reporting

```
[Parsing]
Initializing parsers...               python, javascript, typescript
Parsing files...                      8/8 (100%)
  src/auth/login.py                   12 elements
  src/auth/session.py                 8 elements
  src/utils/helpers.js                5 elements
  ...

Summary:
  Files parsed:    8
  Parse errors:    0
  Total elements:  67
    Classes:       5
    Functions:     23
    Methods:       31
    Variables:     8
```

---

## Error Handling

| Error | Action |
|-------|--------|
| Grammar not available | Error: "Language {lang} not supported" |
| Query file missing | Error: "Query file not found: {path}" |
| Query syntax error | Error: "Invalid query for {lang}: {error}" |
| Parse timeout | Log warning, skip file (set 30s timeout) |
| Memory exhaustion | Log error, skip file, continue |

### Timeout Handling

```python
import signal

def parse_with_timeout(content: bytes, parser: Parser, timeout: int = 30) -> Tree:
    """Parse with timeout to handle pathological cases"""

    def handler(signum, frame):
        raise TimeoutError("Parse timeout")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        return parser.parse(content)
    finally:
        signal.alarm(0)
```

---

## Performance Considerations

| Operation | Bottleneck | Optimization |
|-----------|------------|--------------|
| Grammar loading | Disk I/O | Cache in memory |
| File reading | Disk I/O | Read in parallel (thread pool) |
| Parsing | CPU | Process files in parallel |
| Query execution | CPU | Reuse compiled queries |

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def parse_files_parallel(
    files: List[FileInfo],
    parser_pool: ParserPool,
    workers: int = 4
) -> List[ParsedFile]:
    """Parse files in parallel"""

    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(parse_single_file, f, parser_pool): f
            for f in files
        }

        for future in as_completed(futures):
            file_info = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                log.error(f"Failed to parse {file_info.relative_path}: {e}")

    return results
```

---

## Summary of Decisions

| Decision | Value |
|----------|-------|
| Parser library | Tree-sitter |
| Query language | Tree-sitter S-expressions (.scm files) |
| Parsing approach | Per-file, parallel |
| Hierarchy detection | Position-based containment |
| Element ID format | `scope:repo:user:path:type:name:line` |
| Error handling | Log and continue (fault-tolerant) |
| Timeout | 30 seconds per file |
| Parallel workers | 4 (configurable) |
| File-level element | Yes (virtual, for hierarchy root) |
| Nested class support | Yes (stack-based) |
