; Rust import extraction queries

; =============================================================================
; USE STATEMENTS
; =============================================================================

; use crate::module;
(use_declaration
  argument: (scoped_identifier) @use.path) @use.simple

; use crate::module as alias;
(use_declaration
  argument: (use_as_clause
    path: (scoped_identifier) @use.path
    alias: (identifier) @use.alias)) @use.aliased

; use crate::module::{a, b};
(use_declaration
  argument: (scoped_use_list
    path: (scoped_identifier)? @use.prefix
    list: (use_list
      (identifier) @use.item)*)) @use.list

; use crate::*;
(use_declaration
  argument: (use_wildcard
    (scoped_identifier)? @use.prefix)) @use.wildcard

; pub use crate::module;
(use_declaration
  (visibility_modifier) @pub_use.visibility
  argument: (_) @pub_use.path) @pub_use.def

; =============================================================================
; EXTERN CRATE
; =============================================================================

; extern crate name;
(extern_crate_declaration
  name: (identifier) @extern_crate.name) @extern_crate.def

; extern crate name as alias;
(extern_crate_declaration
  name: (identifier) @extern_crate_alias.name
  alias: (identifier) @extern_crate_alias.alias) @extern_crate_alias.def
