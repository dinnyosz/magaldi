# External Test Repositories

Curated list of small-to-medium open-source repositories for testing Magaldi's parser pipeline across all supported languages. Repos are chosen to be **small enough to parse in minutes, not hours**, while still exercising diverse language patterns.

## Location

Test repos live in `test_repos/` at the project root. This directory is gitignored so cloned repos won't pollute the Magaldi repo, but they're right there for easy `magaldi parse` during development.

```
magaldi/
├── src/
├── tests/
├── test_repos/          # <-- gitignored, clone targets here
│   ├── click/
│   ├── express/
│   ├── zod/
│   └── ...
└── tools/
    ├── clone-test-repos.sh
    └── parse-test-repos.sh
```

## Quick Start

```bash
# Clone all test repos at once
./tools/clone-test-repos.sh

# Parse all cloned repos (scope=test-repo, user=test)
./tools/parse-test-repos.sh

# Or just the smallest ones
./tools/clone-test-repos.sh --tier 1
./tools/parse-test-repos.sh --tier 1

# Parse a single repo
./tools/parse-test-repos.sh click

# Dry run (no DB needed)
./tools/parse-test-repos.sh --dry-run click
```

---

## Repositories by Language

### Python

| Repo | Size | License | Key Patterns |
|------|------|---------|--------------|
| [pallets/click](https://github.com/pallets/click) | ~4 MB | BSD-3 | Decorators (`@click.command`, `@click.option`), custom types, callback chains, context objects |
| [psf/requests](https://github.com/psf/requests) | ~5 MB | Apache-2.0 | Session classes, adapters, hooks, `__init__` re-exports, exception hierarchies, utility modules |
| [encode/httpx](https://github.com/encode/httpx) | ~8 MB | BSD-3 | Async/await, type annotations, dataclass-like models, transport abstractions, context managers |

**What these exercise:** Decorators, class hierarchies, `__init__.py` re-exports, type hints, async patterns, exception classes, module-level constants.

### JavaScript

| Repo | Size | License | Key Patterns |
|------|------|---------|--------------|
| [expressjs/express](https://github.com/expressjs/express) | ~10 MB | MIT | CommonJS (`require`/`module.exports`), prototype inheritance, middleware chains, method chaining |
| [lodash/lodash](https://github.com/lodash/lodash) | ~30 MB | MIT | Standalone utility functions, complex closures, lazy evaluation, currying, internal helper modules |
| [sindresorhus/got](https://github.com/sindresorhus/got) | ~3 MB | MIT | ESM imports/exports, Promises, class-based HTTP client, options merging, error subclasses |

**What these exercise:** CommonJS vs ESM, prototype vs class patterns, closures, higher-order functions, factory functions, callback patterns.

### TypeScript

| Repo | Size | License | Key Patterns |
|------|------|---------|--------------|
| [colinhacks/zod](https://github.com/colinhacks/zod) | ~5 MB | MIT | Complex generics, conditional types, method chaining (fluent API), discriminated unions, type inference |
| [trpc/trpc](https://github.com/trpc/trpc) | ~15 MB | MIT | Monorepo packages, generics with constraints, decorators, middleware patterns, type-safe routers |
| [drizzle-team/drizzle-orm](https://github.com/drizzle-team/drizzle-orm) | ~20 MB | Apache-2.0 | Template literal types, mapped types, builder pattern, SQL DSL, complex re-exports across packages |

**What these exercise:** Advanced generics, conditional/mapped/template literal types, monorepo structure, interfaces, enums, abstract classes, namespace usage.

### PHP

| Repo | Size | License | Key Patterns |
|------|------|---------|--------------|
| [composer/composer](https://github.com/composer/composer) | ~29 MB | MIT | Namespaces, autoloading, command pattern, class hierarchies, JSON schema handling |
| [guzzle/guzzle](https://github.com/guzzle/guzzle) | ~5 MB | MIT | Promises, middleware stack, PSR-7/PSR-18 interfaces, traits, handler pattern |
| [PHPMailer/PHPMailer](https://github.com/PHPMailer/PHPMailer) | ~8 MB | LGPL-2.1 | Single large class, exception handling, constants, static methods, SMTP state machine |

**What these exercise:** Namespaces, traits, interfaces, abstract classes, type hints, PHP 8 features (where used), PSR compliance, large class files.

### Rust

| Repo | Size | License | Key Patterns |
|------|------|---------|--------------|
| [BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep) | ~5 MB | MIT | Workspace with multiple crates, traits, enums, error types, builder pattern, iterators |
| [sharkdp/bat](https://github.com/sharkdp/bat) | ~11 MB | Apache-2.0 | Structs/enums, trait implementations, `clap` derive macros, lazy_static, module organization |
| [sharkdp/fd](https://github.com/sharkdp/fd) | ~3 MB | Apache-2.0 | Clean small crate, CLI arg parsing, thread pools, closures, `Result`/`Option` chains, walkdir patterns |

**What these exercise:** Traits, generics with lifetimes, derive macros, `impl` blocks, `mod` organization, error handling patterns, workspace/multi-crate structure.

### Bash / Shell

| Repo | Size | License | Key Patterns |
|------|------|---------|--------------|
| [nvm-sh/nvm](https://github.com/nvm-sh/nvm) | ~4 MB | MIT | One large script (`nvm.sh`), nested functions, local variables, heredocs, command substitution, arithmetic |
| [dylanaraps/neofetch](https://github.com/dylanaraps/neofetch) | ~1 MB | MIT | Single large bash file, case statements, printf formatting, function-heavy, variable expansion patterns |
| [rbenv/rbenv](https://github.com/rbenv/rbenv) | ~1 MB | MIT | Many small shell scripts, completions, hook system, function definitions, shims pattern |

**What these exercise:** Function definitions, variable scoping (local), heredocs, case/if patterns, command substitution, array usage, sourcing other scripts.

### Polyglot (Multiple Languages)

| Repo | Languages | Size | License | Notes |
|------|-----------|------|---------|-------|
| [nickel-lang/nickel](https://github.com/nickel-lang/nickel) | Rust + JS/TS | ~25 MB | MIT | Rust core + JS/TS WASM bindings. Good for testing multi-language indexing in one repo |

---

## Test Tiers

Use these tiers depending on what you're testing:

### Tier 1: Smoke Test (~1 min each)
Quick validation that parsing works for each language:
- `click` (Python, 4 MB)
- `got` (JavaScript, 3 MB)
- `zod` (TypeScript, 5 MB)
- `guzzle` (PHP, 5 MB)
- `fd` (Rust, 3 MB)
- `neofetch` (Bash, 1 MB)

### Tier 2: Pattern Coverage (~5 min each)
Broader pattern coverage for parser improvement work:
- `httpx` (Python) — async, types
- `express` (JavaScript) — CommonJS, prototypes
- `trpc` (TypeScript) — monorepo, generics
- `composer` (PHP) — namespaces, commands
- `ripgrep` (Rust) — workspace, multi-crate
- `rbenv` (Bash) — many small scripts

### Tier 3: Stress Test (~15-30 min each)
For benchmarking and performance testing (not in the clone script):
- `django/django` (Python, 277 MB)
- `facebook/react` (JS/TS, 938 MB)
- `angular/angular` (TypeScript, 615 MB)
- `symfony/symfony` (PHP, 321 MB)
- `bevyengine/bevy` (Rust, 159 MB)
- `ohmyzsh/ohmyzsh` (Shell, 13 MB)

> Only use Tier 3 for benchmarking. Clone with `--depth 1` and consider `--filter=blob:none` for very large repos.

---

## Tips

- **Always shallow clone** (`--depth 1`) — git history isn't needed for parsing and saves significant disk/time.
- **Sparse checkout** for huge repos — if you only need the `src/` directory:
  ```bash
  git clone --depth 1 --filter=blob:none --sparse https://github.com/org/repo.git
  cd repo
  git sparse-checkout set src/
  ```
- **Track parse times** — use `time magaldi parse .` to compare across repos and after parser changes.
- **Compare element counts** — after parsing, check how many elements were indexed per repo to catch regressions:
  ```bash
  # Quick element count via MCP
  magaldi mcp get_repo_stats
  ```
