; Python import extraction queries

; =============================================================================
; SIMPLE IMPORTS
; =============================================================================

; import module
(import_statement
  name: (dotted_name) @import.module) @import.simple

; import module as alias
(import_statement
  name: (aliased_import
    name: (dotted_name) @import_alias.module
    alias: (identifier) @import_alias.alias)) @import_alias.simple

; =============================================================================
; FROM IMPORTS
; =============================================================================

; from module import name
(import_from_statement
  module_name: (dotted_name) @from.module
  name: (dotted_name) @from.name) @from.simple

; from module import name as alias
(import_from_statement
  module_name: (dotted_name) @from_alias.module
  name: (aliased_import
    name: (dotted_name) @from_alias.name
    alias: (identifier) @from_alias.alias)) @from_alias.stmt

; from module import multiple names
(import_from_statement
  module_name: (dotted_name)? @from_multi.module
  (dotted_name) @from_multi.name) @from_multi.stmt

; =============================================================================
; RELATIVE IMPORTS
; =============================================================================

; from . import name (relative)
(import_from_statement
  module_name: (relative_import) @relative.prefix
  name: (dotted_name) @relative.name) @relative.stmt
