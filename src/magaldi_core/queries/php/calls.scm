; PHP call extraction queries

; =============================================================================
; FUNCTION CALLS
; =============================================================================

; Simple function call: func()
(function_call_expression
  function: (name) @call.name
  arguments: (arguments) @call.args) @call.simple

; Qualified function call: Namespace\func()
(function_call_expression
  function: (qualified_name) @call.qualified_name
  arguments: (arguments) @call.args) @call.qualified

; =============================================================================
; METHOD CALLS
; =============================================================================

; Method call: $obj->method()
(member_call_expression
  object: (_) @call.object
  name: (name) @call.method
  arguments: (arguments) @call.args) @call.method

; Static method call: Class::method()
(scoped_call_expression
  scope: (_) @call.class
  name: (name) @call.static_method
  arguments: (arguments) @call.args) @call.static

; Nullsafe method call: $obj?->method()
(nullsafe_member_call_expression
  object: (_) @call.nullsafe_object
  name: (name) @call.nullsafe_method
  arguments: (arguments) @call.args) @call.nullsafe

; =============================================================================
; CONSTRUCTOR CALLS
; =============================================================================

; new ClassName()
(object_creation_expression
  (name) @new.name
  (arguments)? @new.args) @new.call

; new Qualified\ClassName()
(object_creation_expression
  (qualified_name) @new.qualified_name
  (arguments)? @new.args) @new.qualified

; new $variable()
(object_creation_expression
  (variable_name) @new.variable
  (arguments)? @new.args) @new.dynamic

; =============================================================================
; SPECIAL CALLS
; =============================================================================

; Parent method call: parent::method()
(scoped_call_expression
  scope: (relative_scope) @parent_call.scope
  name: (name) @parent_call.method
  arguments: (arguments) @parent_call.args)
  (#eq? @parent_call.scope "parent") @parent_call.def

; Self method call: self::method()
(scoped_call_expression
  scope: (relative_scope) @self_call.scope
  name: (name) @self_call.method
  arguments: (arguments) @self_call.args)
  (#eq? @self_call.scope "self") @self_call.def

; Closure call: $closure()
(function_call_expression
  function: (variable_name) @closure_call.name
  arguments: (arguments) @closure_call.args) @closure_call.def

; =============================================================================
; CHAINED CALLS
; =============================================================================

; Chained method calls: $obj->a()->b()
(member_call_expression
  object: (member_call_expression) @chain.parent
  name: (name) @chain.method
  arguments: (arguments) @chain.args) @chain.call
