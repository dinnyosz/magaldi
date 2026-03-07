; Java element extraction queries
; Used by parser_lab and extractors for pattern-based code discovery

; =============================================================================
; CLASSES
; =============================================================================

; Class declaration
(class_declaration
  name: (identifier) @class.name
  body: (class_body) @class.body) @class.def

; =============================================================================
; INTERFACES
; =============================================================================

; Interface declaration
(interface_declaration
  name: (identifier) @interface.name
  body: (interface_body) @interface.body) @interface.def

; =============================================================================
; ENUMS
; =============================================================================

; Enum declaration
(enum_declaration
  name: (identifier) @enum.name
  body: (enum_body) @enum.body) @enum.def

; =============================================================================
; RECORDS
; =============================================================================

; Record declaration (Java 16+)
(record_declaration
  name: (identifier) @record.name
  parameters: (formal_parameters) @record.params) @record.def

; =============================================================================
; METHODS
; =============================================================================

; Method declaration
(method_declaration
  name: (identifier) @method.name
  parameters: (formal_parameters) @method.params) @method.def

; Constructor declaration
(constructor_declaration
  name: (identifier) @constructor.name
  parameters: (formal_parameters) @constructor.params
  body: (constructor_body) @constructor.body) @constructor.def

; =============================================================================
; FIELDS
; =============================================================================

; Field declaration
(field_declaration
  type: (_) @field.type
  declarator: (variable_declarator
    name: (identifier) @field.name)) @field.def

; =============================================================================
; ANNOTATIONS
; =============================================================================

; Annotation
(annotation
  name: (identifier) @annotation.name) @annotation.def

; Marker annotation (shorthand for annotation without args)
(marker_annotation
  name: (identifier) @marker_annotation.name) @marker_annotation.def

; =============================================================================
; ANNOTATION TYPES
; =============================================================================

; Annotation type declaration (@interface)
(annotation_type_declaration
  name: (identifier) @annotation_type.name) @annotation_type.def

; =============================================================================
; CONSTANTS
; =============================================================================

; Enum constant
(enum_constant
  name: (identifier) @enum_constant.name) @enum_constant.def
