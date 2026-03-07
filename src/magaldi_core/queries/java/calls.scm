; Java call extraction queries

; =============================================================================
; METHOD INVOCATIONS
; =============================================================================

; Simple method call: method()
(method_invocation
  name: (identifier) @call.name
  arguments: (argument_list) @call.args) @call.simple

; Qualified method call: obj.method()
(method_invocation
  object: (identifier) @call.receiver
  name: (identifier) @call.method
  arguments: (argument_list) @call.method_args) @call.qualified

; This method call: this.method()
(method_invocation
  object: (this) @call.this
  name: (identifier) @call.this_method
  arguments: (argument_list) @call.this_args) @call.this_call

; Super method call: super.method()
(method_invocation
  object: (super) @call.super
  name: (identifier) @call.super_method
  arguments: (argument_list) @call.super_args) @call.super_call

; Chained method call: obj.a().b()
(method_invocation
  object: (method_invocation) @chain.parent
  name: (identifier) @chain.method
  arguments: (argument_list) @chain.args) @chain.call

; =============================================================================
; CONSTRUCTOR CALLS
; =============================================================================

; new ClassName()
(object_creation_expression
  type: (type_identifier) @new.type
  arguments: (argument_list) @new.args) @new.call

; new ClassName<T>()
(object_creation_expression
  type: (generic_type
    (type_identifier) @new.generic_type)
  arguments: (argument_list) @new.generic_args) @new.generic_call

; =============================================================================
; STATIC / FIELD ACCESS CALLS
; =============================================================================

; ClassName.staticMethod() via field_access
(method_invocation
  object: (field_access) @static_call.receiver
  name: (identifier) @static_call.method
  arguments: (argument_list) @static_call.args) @static_call.call

; =============================================================================
; LAMBDA AND METHOD REFERENCES
; =============================================================================

; Lambda expression: (x) -> expr
(lambda_expression
  parameters: (_) @lambda.params
  body: (_) @lambda.body) @lambda.def

; Method reference: ClassName::method
(method_reference) @method_ref.def
