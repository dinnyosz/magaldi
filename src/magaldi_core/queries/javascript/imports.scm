; JavaScript/TypeScript import extraction queries

; =============================================================================
; ES6 IMPORTS
; =============================================================================

; import x from 'module'
(import_statement
  (import_clause
    (identifier) @import.default)
  source: (string) @import.source) @import.default_import

; import { x } from 'module'
(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @import.name
        alias: (identifier)? @import.alias)))
  source: (string) @import.source) @import.named

; import * as x from 'module'
(import_statement
  (import_clause
    (namespace_import
      (identifier) @import.namespace))
  source: (string) @import.source) @import.namespace_import

; import 'module' (side-effect)
(import_statement
  source: (string) @import.source) @import.side_effect

; =============================================================================
; DYNAMIC IMPORTS
; =============================================================================

; import('module')
(call_expression
  function: (import)
  arguments: (arguments
    (string) @dynamic_import.source)) @dynamic_import.call

; =============================================================================
; COMMONJS REQUIRES
; =============================================================================

; const x = require('module')
(lexical_declaration
  (variable_declarator
    name: (identifier) @require.name
    value: (call_expression
      function: (identifier) @require.func
      arguments: (arguments
        (string) @require.source))))
  (#eq? @require.func "require") @require.def

; const { x } = require('module')
(lexical_declaration
  (variable_declarator
    name: (object_pattern) @require.destructure
    value: (call_expression
      function: (identifier) @require.func
      arguments: (arguments
        (string) @require.source))))
  (#eq? @require.func "require") @require.destructure_def

; =============================================================================
; TYPE IMPORTS (TypeScript)
; =============================================================================

; import type { x } from 'module'
(import_statement
  "type" @type_import.type
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @type_import.name)))
  source: (string) @type_import.source) @type_import.def
