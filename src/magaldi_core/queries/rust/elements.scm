; Rust element extraction queries
; Used by parser_lab and extractors for pattern-based code discovery

; =============================================================================
; FUNCTIONS
; =============================================================================

; Function definition
(function_item
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body) @function.def

; =============================================================================
; STRUCTS
; =============================================================================

; Struct definition
(struct_item
  name: (type_identifier) @struct.name) @struct.def

; =============================================================================
; ENUMS
; =============================================================================

; Enum definition
(enum_item
  name: (type_identifier) @enum.name
  body: (enum_variant_list) @enum.variants) @enum.def

; =============================================================================
; IMPL BLOCKS
; =============================================================================

; Impl block for type
(impl_item
  type: (type_identifier) @impl.type
  body: (declaration_list) @impl.body) @impl.def

; =============================================================================
; TRAITS
; =============================================================================

; Trait definition
(trait_item
  name: (type_identifier) @trait.name
  body: (declaration_list) @trait.body) @trait.def

; =============================================================================
; MODULES
; =============================================================================

; Module definition
(mod_item
  name: (identifier) @module.name) @module.def

; =============================================================================
; CONSTANTS & STATICS
; =============================================================================

; Const definition
(const_item
  name: (identifier) @const.name) @const.def

; Static definition
(static_item
  name: (identifier) @static.name) @static.def

; =============================================================================
; TYPE ALIASES
; =============================================================================

; Type alias
(type_item
  name: (type_identifier) @type_alias.name) @type_alias.def

; =============================================================================
; MACROS
; =============================================================================

; Declarative macro
(macro_definition
  name: (identifier) @macro.name) @macro.def
