; PHP import extraction queries

; =============================================================================
; USE STATEMENTS
; =============================================================================

; use Namespace\Class;
(namespace_use_declaration
  (namespace_use_clause
    (qualified_name) @use.name)) @use.def

; use Namespace\Class as Alias;
(namespace_use_declaration
  (namespace_use_clause
    (qualified_name) @use_alias.name
    (namespace_aliasing_clause
      (name) @use_alias.alias))) @use_alias.def

; use function Namespace\func;
(namespace_use_declaration
  "function" @use_func.type
  (namespace_use_clause
    (qualified_name) @use_func.name)) @use_func.def

; use const Namespace\CONST;
(namespace_use_declaration
  "const" @use_const.type
  (namespace_use_clause
    (qualified_name) @use_const.name)) @use_const.def

; Grouped use: use Namespace\{A, B, C};
(namespace_use_declaration
  (namespace_use_group
    (namespace_name) @use_group.prefix
    (namespace_use_group_clause
      (namespace_name) @use_group.name)*)) @use_group.def

; =============================================================================
; INCLUDES/REQUIRES
; =============================================================================

; require/require_once
(expression_statement
  (require_expression
    (string) @require.path)) @require.def

(expression_statement
  (require_once_expression
    (string) @require_once.path)) @require_once.def

; include/include_once
(expression_statement
  (include_expression
    (string) @include.path)) @include.def

(expression_statement
  (include_once_expression
    (string) @include_once.path)) @include_once.def
