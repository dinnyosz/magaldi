"""Configuration and skill generation tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _detect_mcp_installation_scope(project_root: str | None) -> str:
    """Detect whether magaldi MCP is installed at project or global level.

    Checks for magaldi in MCP server configs to determine default scope.

    Returns:
        "project" if found in project config, "global" otherwise.
    """
    # Check project-level config first
    if project_root:
        project_claude_json = Path(project_root) / ".claude.json"
        if project_claude_json.exists():
            try:
                config = json.loads(project_claude_json.read_text())
                mcp_servers = config.get("mcpServers", {})
                if "magaldi" in mcp_servers:
                    return "project"
            except (json.JSONDecodeError, OSError):
                pass

    # Check global config
    global_claude_json = Path.home() / ".claude.json"
    if global_claude_json.exists():
        try:
            config = json.loads(global_claude_json.read_text())
            mcp_servers = config.get("mcpServers", {})
            if "magaldi" in mcp_servers:
                return "global"
        except (json.JSONDecodeError, OSError):
            pass

    # Default to project if can't determine
    return "project"


# Skill content constant
_SKILL_CONTENT = """---
name: magaldi
description: >
  ALWAYS use for: grep, find usages, search patterns, find implementations,
  call graphs, find where X is used/called, search code by meaning.
  These tools use the PRE-INDEXED codebase for faster, richer results than raw file search.
  Invoke BEFORE using built-in Grep/Glob/Read tools.
---

# Magaldi Code Discovery

**CRITICAL: Use magaldi tools INSTEAD OF built-in Grep/Glob for code search.**

## REQUIRED: Read magaldi.yaml First

**BEFORE using any magaldi tool that requires scope/repository parameters, you MUST read `magaldi.yaml` from the repository root.**

```yaml
# magaldi.yaml (in repository root)
scope: myorg        # Use this for scope parameter
repository: myrepo  # Use this for repository parameter
```

Do NOT guess or hardcode these values. Read the file.

If `magaldi.yaml` doesn't exist, create it:
```
mcp__magaldi__generate_config(repo_path="/path/to/repo")
```

## Getting Started

Check `CLAUDE.md` for project-specific guidance including architecture, development commands, and configuration details.

You can search for tools to: search code semantically, find files by pattern,
grep/regex search, find where functions are called, trace call chains,
analyze dependencies and imports, detect design patterns, find dead code,
audit security issues, find complex functions, list HTTP routes and CLI commands,
explore glossary terms, and inspect code elements with AI summaries.

## What's Pre-indexed

The codebase is pre-indexed with:
- Semantic embeddings (search by meaning)
- Pre-computed summaries (understand without reading)
- Call graphs (who calls what)
- Feature clustering (related functions grouped)

## When to Use Magaldi vs Built-in Tools

| User Request | USE THIS | NOT THIS |
|--------------|----------|----------|
| "grep for X" / "find pattern X" | `mcp__magaldi__pattern_search` (mode="regexp") | Built-in Grep |
| "find where X is used/called" | `mcp__magaldi__find_usages` | Built-in Grep |
| "search for functions that do X" | `mcp__magaldi__search_code` | Built-in Grep |
| "find files matching *.py" | `mcp__magaldi__find_files` | Built-in Glob |
| "what implements Interface X" | `mcp__magaldi__find_implementations` | Built-in Grep |
| "who calls this function" | `mcp__magaldi__get_call_graph` | Built-in Grep |
| "find similar code to X" | `mcp__magaldi__find_similar` | N/A |
| "what does the codebase do" | `mcp__magaldi__search_features` | N/A |

## Why Magaldi Tools Are Better

| Feature | Magaldi | Built-in Grep/Glob |
|---------|---------|-------------------|
| Pre-indexed | Yes - instant results | No - scans every file |
| Summaries | Every function has AI summary | None |
| Semantic search | "authentication" finds login, auth, verify | Only literal matches |
| Call graphs | Built-in | Must grep manually |
| Context | Parent class, siblings, children | Just file/line |

## Tool Priority (Use in This Order)

### 1. SEMANTIC SEARCH (Start Here for "what does X do")
```
mcp__magaldi__search_code(query="authentication logic", brief=true)
```
- Natural language: "function that validates tokens"
- Returns summaries, not just file:line
- Use `brief=true` for exploration

### 2. PATTERN SEARCH (For literal patterns, regex, wildcards)
```
mcp__magaldi__pattern_search(pattern="add_job.*\\\\(", mode="regexp", scope="...", repository="...")
```
- **Three modes:**
  - `regexp`: Lucene regex (e.g., `"add_column.*Model"`)
  - `wildcard`: Simple wildcards (e.g., `"*column*Model*"`)
  - `proximity`: Terms near each other (e.g., `"add column Model"` with slop=5)
- ES-native - queries run server-side for better performance
- Requires `scope` and `repository` parameters

### 3. USAGE TRACKING (For "where is X called")
```
mcp__magaldi__find_usages(element_id="...")
```
- After search_code found the element
- Shows all call sites with context
- Filters out definitions automatically

### 4. RELATIONSHIPS (For refactoring, impact analysis)
```
mcp__magaldi__get_call_graph(element_id="...")
mcp__magaldi__find_implementations(class_name="BaseClass")
```
- Before modifying shared code
- Understanding dependencies

## Workflow Examples

### "Grep for X" / "Find pattern X"
```
1. mcp__magaldi__pattern_search(pattern="X", mode="regexp", scope="...", repository="...")
   - NOT: built-in Grep tool
   - For wildcards: mode="wildcard" with patterns like "*X*"
   - For proximity: mode="proximity" with slop parameter
```

### "Find where function X is called"
```
1. mcp__magaldi__search_code(query="X", element_types=["function"])
2. mcp__magaldi__find_usages(element_id=result.element_id)
   - NOT: grep for "X("
```

### "What implements interface Y"
```
1. mcp__magaldi__find_implementations(class_name="Y")
   - NOT: grep for "class.*Y"
```

### "How does X work"
```
1. mcp__magaldi__search_code(query="X functionality", brief=true)
2. mcp__magaldi__get_element(element_id=best_match, include_code=true)
   - NOT: grep then read file
```

### "Find all authentication code"
```
1. mcp__magaldi__search_features(query="authentication")
2. mcp__magaldi__get_feature_members(feature_id=result.feature_id)
   - Returns grouped, related functions
```

### "Refactor function Z"
```
1. mcp__magaldi__search_code(query="Z")
2. mcp__magaldi__find_usages(element_id)  # Impact analysis
3. mcp__magaldi__get_call_graph(element_id)  # Dependencies
4. THEN make changes
```

## Deferred Tool Loading (API Users)

When using Magaldi via the Anthropic API with `mcp_toolset`, configure deferred loading to save context tokens. Default all tools to deferred, keep the top 5 most-used always loaded:

```json
{
  "type": "mcp_toolset",
  "mcp_server_name": "magaldi",
  "default_config": { "defer_loading": true },
  "configs": {
    "search_code": { "defer_loading": false },
    "find_files": { "defer_loading": false },
    "get_element": { "defer_loading": false },
    "pattern_search": { "defer_loading": false },
    "get_file_structure": { "defer_loading": false }
  }
}
```

Claude Code users: deferred loading is automatic when tool descriptions exceed 10K tokens (Magaldi qualifies).

## Anti-Patterns (NEVER Do These)

1. **Using built-in Grep instead of magaldi__pattern_search**
   - Magaldi pattern_search runs queries server-side in OpenSearch
   - Built-in Grep scans files one by one

2. **Using built-in Glob instead of magaldi__find_files**
   - Magaldi knows which files are indexed

3. **Grepping for function calls instead of find_usages**
   - find_usages filters definitions, has context

4. **Reading whole files to understand them**
   - Use search_code -> get_element with summaries

5. **Skipping semantic search**
   - Summaries save tokens, embeddings find related code

## Available Tools Quick Reference

| Tool | Purpose |
|------|---------|
| `search_code` | Semantic search by meaning |
| `search_features` | Find high-level capabilities |
| `pattern_search` | **ES-native pattern search** - regexp, wildcard, or proximity mode |
| `find_usages` | Where is this called/used (uses ES regexp internally) |
| `find_implementations` | What implements this interface (uses ES regexp internally) |
| `get_call_graph` | Callers and callees |
| `find_similar` | Similar code patterns |
| `get_element` | Full element details |
| `get_context` | Parent, siblings, children |
| `find_files` | Glob pattern search (USE THIS not built-in Glob) |
| `list_features` | All features in repo |
| `get_feature_members` | Functions in a feature |
| `list_repos` | All indexed repos |
| `get_repo_stats` | Repository statistics |

## Token-Saving Parameters

High-volume tools support `max_tokens` and `filename` for output control:

```
# Limit response size (drops trailing results to fit budget)
mcp__magaldi__search_code(query="auth", max_tokens=500)

# Save full results to file, get summary inline
mcp__magaldi__find_dead_code(filename="/tmp/dead_code.md")
```

**Supported tools:** search_code, search_features, find_usages, pattern_search,
find_callers, find_call_chain, explain_element, find_dead_code, find_entry_points.

## Subagent Delegation

**Delegate to Explore subagent when:**
- Multi-step workflows (search -> inspect -> find_usages)
- Exploratory queries with unknown result count
- Results won't be directly referenced in the next action

**Call inline when:**
- Single-call lookups with known hash_id (get_element, get_context)
- Quick searches expected to return <5 results
- Results needed for the immediate next action

## Remember

The index has already done the hard work:
- Code is parsed and structured
- Summaries explain what code does
- Embeddings enable semantic search
- Call graphs are pre-computed

**Use magaldi tools. Don't re-grep what's already indexed.**
"""

# List of all magaldi tools
_ALL_TOOLS = [
    "mcp__magaldi__search_code",
    "mcp__magaldi__search_features",
    "mcp__magaldi__find_similar",
    "mcp__magaldi__pattern_search",
    "mcp__magaldi__find_usages",
    "mcp__magaldi__find_implementations",
    "mcp__magaldi__find_files",
    "mcp__magaldi__get_file_structure",
    "mcp__magaldi__get_element",
    "mcp__magaldi__batch_get_elements",
    "mcp__magaldi__get_context",
    "mcp__magaldi__get_children",
    "mcp__magaldi__get_call_graph",
    "mcp__magaldi__find_callers",
    "mcp__magaldi__find_call_chain",
    "mcp__magaldi__list_features",
    "mcp__magaldi__get_feature_members",
    "mcp__magaldi__list_glossary",
    "mcp__magaldi__get_glossary_term",
    "mcp__magaldi__search_glossary",
    "mcp__magaldi__list_repos",
    "mcp__magaldi__get_repo_stats",
    "mcp__magaldi__find_dead_code",
    "mcp__magaldi__find_entry_points",
    "mcp__magaldi__get_route_tree",
    "mcp__magaldi__get_command_tree",
    "mcp__magaldi__find_dependencies",
    "mcp__magaldi__find_dependents",
    "mcp__magaldi__dependency_graph",
    "mcp__magaldi__list_patterns",
    "mcp__magaldi__find_by_pattern",
    "mcp__magaldi__find_complex_functions",
    "mcp__magaldi__find_security_issues",
    "mcp__magaldi__find_undocumented",
    "mcp__magaldi__find_env_usage",
    "mcp__magaldi__find_async_code",
    "mcp__magaldi__explain_element",
    "mcp__magaldi__generate_skill",
    "mcp__magaldi__generate_config",
]


def generate_skill(
    project_root: str | None = None,
    skill_name: str = "magaldi",
    scope: str | None = None,
    add_allowed_tools: bool = False,
) -> dict[str, Any]:
    """Generate a SKILL.md file that teaches LLMs how to use this MCP effectively.

    Args:
        project_root: Project root directory (required for scope="project").
        skill_name: Name of the skill (default: "magaldi").
        scope: Where to install - "project" or "global". Defaults to MCP installation scope.
        add_allowed_tools: If True, add all magaldi tools to allowed tools in settings.

    Returns:
        Dict with skill content and metadata.
    """
    # Auto-detect scope based on MCP installation if not specified
    if scope is None:
        scope = _detect_mcp_installation_scope(project_root)
        detected_scope = True
    else:
        detected_scope = False

    result: dict[str, Any] = {
        "skill_name": skill_name,
        "content": _SKILL_CONTENT,
        "version": "1.0.0",
        "scope": scope,
    }
    if detected_scope:
        result["scope_auto_detected"] = True
        result["scope_note"] = f"Auto-detected '{scope}' based on MCP installation location"

    # Determine target path based on scope
    if scope == "global":
        skill_dir = Path.home() / ".claude" / "skills" / skill_name
        skill_path = skill_dir / "SKILL.md"
    elif scope == "project":
        if not project_root:
            result["error"] = "project_root is required for scope='project'"
            return result
        skill_dir = Path(project_root) / ".claude" / "skills" / skill_name
        skill_path = skill_dir / "SKILL.md"
    else:
        result["error"] = f"Invalid scope '{scope}'. Use 'project' or 'global'."
        return result

    # Check for existing skill in both locations to avoid duplication
    global_path = Path.home() / ".claude" / "skills" / skill_name / "SKILL.md"
    project_path = (
        Path(project_root) / ".claude" / "skills" / skill_name / "SKILL.md"
        if project_root
        else None
    )

    skill_already_exists = skill_path.exists()
    if skill_already_exists:
        result["skipped"] = True
        result["reason"] = f"Skill already exists at: {skill_path}"
        result["path"] = str(skill_path)
        # Don't return early - still update CLAUDE.md if needed

    # Warn if exists in the other location
    if scope == "project" and global_path.exists():
        result["warning"] = f"Note: Skill also exists globally at {global_path}"
    elif scope == "global" and project_path and project_path.exists():
        result["warning"] = f"Note: Skill also exists in project at {project_path}"

    # Write the skill file (only if it doesn't already exist)
    if not skill_already_exists:
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(_SKILL_CONTENT)
        result["written_to"] = str(skill_path)

    # Add allowed tools to settings if requested
    if add_allowed_tools:
        # Determine settings path based on scope
        if scope == "global":
            settings_path = Path.home() / ".claude" / "settings.json"
        else:
            settings_path = Path(project_root) / ".claude" / "settings.json"

        # Load existing settings or create new
        settings: dict[str, Any] = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
            except json.JSONDecodeError:
                settings = {}

        # Add tools to allowedTools (use wildcard for simplicity)
        existing_allowed = settings.get("allowedTools", [])
        wildcard_tool = "mcp__magaldi__*"
        if wildcard_tool not in existing_allowed:
            existing_allowed.append(wildcard_tool)
            settings["allowedTools"] = existing_allowed

            # Write settings
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(json.dumps(settings, indent=2))
            result["allowed_tools_added"] = True
            result["settings_updated"] = str(settings_path)
        else:
            result["allowed_tools_added"] = False
            result["allowed_tools_note"] = "Magaldi tools already in allowedTools"

    # Add reference to CLAUDE.md (auto-detect project_root from cwd if not provided)
    effective_project_root = project_root or str(Path.cwd())
    if effective_project_root:
        claude_md_path = Path(effective_project_root) / "CLAUDE.md"
        skill_reference = "## Magaldi MCP\n\n**See `.claude/skills/magaldi/SKILL.md` for detailed MCP tool usage guidance.**\n\nMagaldi tools auto-detect `scope` and `repository` from `magaldi.yaml` - no need to specify these parameters manually.\n"

        if claude_md_path.exists():
            content = claude_md_path.read_text()
            # Check if reference already exists
            if ".claude/skills/magaldi/SKILL.md" not in content:
                # Append to end of file
                updated_content = content.rstrip() + "\n\n" + skill_reference
                claude_md_path.write_text(updated_content)
                result["claude_md_updated"] = True
            else:
                result["claude_md_updated"] = False
                result["claude_md_note"] = "Magaldi skill reference already exists"
        else:
            # Create CLAUDE.md with skill reference
            claude_md_path.write_text(f"# Project\n\n{skill_reference}")
            result["claude_md_created"] = True

    # Provide next steps
    next_steps = ["Restart Claude Code to pick up the new skill."]
    if not add_allowed_tools:
        next_steps.append(
            "To allow all magaldi tools without prompts, call generate_skill again with add_allowed_tools=true"
        )
    result["next_steps"] = next_steps
    result["available_tools"] = _ALL_TOOLS

    return result


def generate_config(
    repo_path: str,
    scope: str | None = None,
    repository: str | None = None,
) -> dict[str, Any]:
    """Generate a magaldi.yaml config file for a repository.

    Auto-detects repository name from directory and scope from parent directory.
    Creates config at repo_path/magaldi.yaml.

    Args:
        repo_path: Path to the repository root directory.
        scope: Override auto-detected scope (optional).
        repository: Override auto-detected repository name (optional).

    Returns:
        Dict with config details and path.
    """
    repo_path_obj = Path(repo_path).resolve()

    # Validate path
    if not repo_path_obj.exists():
        return {"error": f"Path does not exist: {repo_path}"}
    if not repo_path_obj.is_dir():
        return {"error": f"Path is not a directory: {repo_path}"}

    # Auto-detect repository name from directory name
    detected_repository = repo_path_obj.name

    # Auto-detect scope from parent directory name
    # Common patterns: /Users/username/code/repo -> scope=username
    # or /home/username/projects/repo -> scope=username
    parent = repo_path_obj.parent
    detected_scope = parent.name

    # Use overrides if provided
    final_scope = scope or detected_scope
    final_repository = repository or detected_repository

    config_path = repo_path_obj / "magaldi.yaml"

    # Check if config already exists
    if config_path.exists():
        return {
            "skipped": True,
            "reason": f"Config already exists at: {config_path}",
            "path": str(config_path),
        }

    # Build config content
    config_data = {
        "scope": final_scope,
        "repository": final_repository,
    }

    # Write config file
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"# Magaldi configuration for {final_repository}\n")
        f.write("# Scope groups related repositories (e.g., org name, username)\n\n")
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    return {
        "scope": final_scope,
        "repository": final_repository,
        "path": str(config_path),
        "auto_detected": {
            "scope": detected_scope,
            "repository": detected_repository,
        },
        "message": f"Created magaldi.yaml at {config_path}",
    }
