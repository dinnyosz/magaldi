; PHP element extraction queries
; Used by parser_lab and extractors for pattern-based code discovery

; =============================================================================
; FUNCTIONS
; =============================================================================

; Function definition
(function_definition
  (name) @function.name
  (formal_parameters) @function.params
  (compound_statement) @function.body) @function.def

; =============================================================================
; CLASSES
; =============================================================================

; Class declaration
(class_declaration
  (name) @class.name
  (declaration_list) @class.body) @class.def

; =============================================================================
; INTERFACES
; =============================================================================

; Interface declaration
(interface_declaration
  (name) @interface.name
  (declaration_list) @interface.body) @interface.def

; =============================================================================
; TRAITS
; =============================================================================

; Trait declaration
(trait_declaration
  (name) @trait.name
  (declaration_list) @trait.body) @trait.def

; =============================================================================
; CLASS MEMBERS
; =============================================================================

; Method definition
(method_declaration
  (name) @method.name
  (formal_parameters) @method.params) @method.def

; Property declaration
(property_declaration
  (property_element
    (variable_name) @property.name)) @property.def

; Constant declaration (class constant)
(const_declaration
  (const_element
    (name) @const.name)) @const.def

; =============================================================================
; NAMESPACES
; =============================================================================

; Namespace definition
(namespace_definition
  (namespace_name) @namespace.name) @namespace.def
