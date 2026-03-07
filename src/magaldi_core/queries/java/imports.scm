; Java import extraction queries

; =============================================================================
; IMPORT DECLARATIONS
; =============================================================================

; Regular import: import java.util.List;
(import_declaration
  (scoped_identifier) @import.path) @import.regular

; Static import: import static java.util.Arrays.asList;
(import_declaration
  "static" @import.static_keyword
  (scoped_identifier) @import.static_path) @import.static

; Wildcard import: import java.util.*;
(import_declaration
  (asterisk) @import.wildcard) @import.wildcard_decl

; =============================================================================
; PACKAGE DECLARATION
; =============================================================================

; package com.example.foo;
(package_declaration
  (scoped_identifier) @package.name) @package.def
