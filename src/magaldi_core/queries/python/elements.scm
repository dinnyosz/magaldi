; Python element extraction queries
; Used by parser_lab and extractors for pattern-based code discovery

; =============================================================================
; FUNCTIONS
; =============================================================================

; Standalone function definition
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  return_type: (type)? @function.return_type
  body: (block) @function.body) @function.def

; Async function definition
(function_definition
  "async" @function.async
  name: (identifier) @async_function.name
  parameters: (parameters) @async_function.params
  return_type: (type)? @async_function.return_type
  body: (block) @async_function.body) @async_function.def

; =============================================================================
; CLASSES
; =============================================================================

; Class definition
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block) @class.body) @class.def

; =============================================================================
; DECORATORS
; =============================================================================

; Decorated definition (function or class)
(decorated_definition
  (decorator)+ @decorated.decorators
  definition: [
    (function_definition) @decorated.function
    (class_definition) @decorated.class
  ]) @decorated.def

; Single decorator - simple identifier
(decorator
  (identifier) @decorator.simple_name) @decorator.simple

; Single decorator - attribute access (e.g., @app.route)
(decorator
  (attribute) @decorator.attr_name) @decorator.attr

; Single decorator - call (e.g., @decorator(args))
(decorator
  (call
    function: [
      (identifier) @decorator.call_name
      (attribute) @decorator.call_attr
    ]
    arguments: (argument_list) @decorator.call_args)) @decorator.call

; =============================================================================
; ASSIGNMENTS (module-level variables/constants)
; =============================================================================

; Module-level assignment
(module
  (expression_statement
    (assignment
      left: (identifier) @assignment.name
      right: (_) @assignment.value))) @assignment.module_level

; Annotated assignment (type hint)
(module
  (expression_statement
    (assignment
      left: (identifier) @typed_assignment.name
      type: (type) @typed_assignment.type
      right: (_)? @typed_assignment.value))) @typed_assignment.module_level

; =============================================================================
; CLASS MEMBERS
; =============================================================================

; Method inside class
(class_definition
  body: (block
    (function_definition
      name: (identifier) @method.name
      parameters: (parameters) @method.params
      return_type: (type)? @method.return_type) @method.def)) @method.class

; Decorated method inside class
(class_definition
  body: (block
    (decorated_definition
      (decorator)+ @decorated_method.decorators
      definition: (function_definition
        name: (identifier) @decorated_method.name
        parameters: (parameters) @decorated_method.params
        return_type: (type)? @decorated_method.return_type) @decorated_method.def))) @decorated_method.class

; Class variable assignment
(class_definition
  body: (block
    (expression_statement
      (assignment
        left: (identifier) @class_var.name
        right: (_) @class_var.value)))) @class_var.class

; =============================================================================
; NESTED FUNCTIONS
; =============================================================================

; Nested function inside function
(function_definition
  body: (block
    (function_definition
      name: (identifier) @nested.name) @nested.def)) @nested.parent
