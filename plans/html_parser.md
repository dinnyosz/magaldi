# HTML Parser Plan

## Goal
Add HTML file support to Magaldi. HTML files are parsed as file-level elements (like markdown/plaintext), but the raw HTML is converted to markdown before being stored as `raw_code`. This dramatically reduces token count for summarization while preserving the semantic content.

## Approach
Follow the same pattern as the CSS/SCSS commit (`cea3616`): discovery → parser → processing integration.

HTML files will NOT use LLM summarization — they'll use the existing markdown parser's section extraction for `document_sections` metadata, plus the converted markdown as `raw_code`. The LLM will summarize the markdown content (not raw HTML), making summaries much cheaper and higher quality.

## Library Choice
Use `markdownify` (BeautifulSoup4-based) — most flexible, handles real-world HTML well, easy to customize tag handling. Alternative: `html2text` (simpler, zero deps beyond stdlib).

## Steps

### Step 1: Add dependency
- Add `markdownify` to `pyproject.toml`
- `pip install -e .`

### Step 2: Create HTML parser (`src/magaldi_core/parsers/html.py`)
- New `HtmlParser` class (does NOT extend `TreeSitterParser` — like `PlainTextParser`)
- `parse()` method:
  1. Convert HTML → markdown using `markdownify`
  2. Create file-level `CodeElement` with `language="html"`
  3. Store **converted markdown** as `raw_code` (not original HTML)
  4. Use existing `MarkdownExtractor` to extract `document_sections` from the converted markdown
- Strip `<script>` and `<style>` tags before conversion (noise for summarization)

### Step 3: Register in discovery + parser registry
- `discovery.py`: Add `.html`, `.htm` → `"html"` to `SUPPORTED_EXTENSIONS`
- `parsers/__init__.py`: Import and export `HtmlParser`
- `code_parser.py`: Add `"html": HtmlParser()` to `PARSERS` dict, update type annotation, import

### Step 4: Processing integration
- No special handling needed in `helpers.py` — HTML files are regular file elements that go through normal LLM summarization
- The key win: `raw_code` is markdown (small) instead of HTML (bloated with tags), so LLM prompts are much smaller
- `should_embed` already includes `"file"` type — no change needed

### Step 5: Tests
- `tests/test_parser_html.py`:
  - Basic HTML → file element with markdown raw_code
  - Script/style tag stripping
  - Document sections extraction from converted markdown
  - Empty/malformed HTML handling
  - `.htm` extension support
  - Large HTML with deeply nested tags

## Key Design Decision
**Store markdown as `raw_code`, not original HTML.** This means:
- ✅ Summarization prompt is 3-10x smaller (no HTML tags)
- ✅ Embedding quality improves (markdown is closer to natural language)
- ✅ Code search finds semantic content, not HTML boilerplate
- ⚠️ Original HTML is not preserved — but for discovery purposes, the semantic content is what matters
