; JavaScript/TypeScript element extraction queries
; Used by parser_lab and extractors for pattern-based code discovery

; =============================================================================
; FUNCTIONS
; =============================================================================

; Function declaration
(function_declaration
  name: (identifier) @function.name
  parameters: (formal_parameters) @function.params
  body: (statement_block) @function.body) @function.def

; Async function declaration
(function_declaration
  "async" @function.async
  name: (identifier) @async_function.name
  parameters: (formal_parameters) @async_function.params
  body: (statement_block) @async_function.body) @async_function.def

; Generator function declaration
(generator_function_declaration
  name: (identifier) @generator.name
  parameters: (formal_parameters) @generator.params
  body: (statement_block) @generator.body) @generator.def

; =============================================================================
; ARROW FUNCTIONS
; =============================================================================

; Arrow function assigned to const/let
(lexical_declaration
  (variable_declarator
    name: (identifier) @arrow.name
    value: (arrow_function
      parameters: [
        (formal_parameters) @arrow.params
        (identifier) @arrow.single_param
      ]?
      body: [
        (statement_block) @arrow.body
        (_) @arrow.expression
      ]))) @arrow.def

; Arrow function assigned to var
(variable_declaration
  (variable_declarator
    name: (identifier) @var_arrow.name
    value: (arrow_function) @var_arrow.value)) @var_arrow.def

; =============================================================================
; CLASSES
; =============================================================================

; Class declaration
(class_declaration
  name: (identifier) @class.name
  (class_heritage
    (identifier) @class.extends)?
  body: (class_body) @class.body) @class.def

; Class expression (const MyClass = class { ... })
(lexical_declaration
  (variable_declarator
    name: (identifier) @class_expr.name
    value: (class
      body: (class_body) @class_expr.body))) @class_expr.def

; =============================================================================
; CLASS MEMBERS
; =============================================================================

; Method definition in class
(class_body
  (method_definition
    name: [
      (property_identifier) @method.name
      (private_property_identifier) @method.private_name
    ]
    parameters: (formal_parameters) @method.params
    body: (statement_block) @method.body)) @method.def

; Static method
(class_body
  (method_definition
    "static" @static_method.static
    name: (property_identifier) @static_method.name
    parameters: (formal_parameters) @static_method.params
    body: (statement_block) @static_method.body)) @static_method.def

; Getter/setter
(class_body
  (method_definition
    ["get" "set"] @accessor.type
    name: (property_identifier) @accessor.name
    parameters: (formal_parameters)? @accessor.params
    body: (statement_block) @accessor.body)) @accessor.def

; Public field (JavaScript)
(class_body
  (field_definition
    property: (property_identifier) @field.name
    value: (_)? @field.value)) @field.def

; Private field (JavaScript)
(class_body
  (field_definition
    property: (private_property_identifier) @private_field.name
    value: (_)? @private_field.value)) @private_field.def

; Public field (TypeScript) — includes static arrow function properties
(class_body
  (public_field_definition
    name: (property_identifier) @ts_field.name
    value: (_)? @ts_field.value)) @ts_field.def

; =============================================================================
; EXPORTS
; =============================================================================

; Export function
(export_statement
  declaration: (function_declaration
    name: (identifier) @export_func.name)) @export_func.def

; Export class
(export_statement
  declaration: (class_declaration
    name: (identifier) @export_class.name)) @export_class.def

; Default export function
(export_statement
  "default" @default_export.default
  declaration: (function_declaration
    name: (identifier)? @default_export.name)) @default_export.def

; =============================================================================
; MODULE-LEVEL VARIABLES
; =============================================================================

; const/let at module level
(program
  (lexical_declaration
    kind: ["const" "let"] @const.kind
    (variable_declarator
      name: (identifier) @const.name
      value: (_) @const.value))) @const.def

; var at module level
(program
  (variable_declaration
    (variable_declarator
      name: (identifier) @var.name
      value: (_)? @var.value))) @var.def
