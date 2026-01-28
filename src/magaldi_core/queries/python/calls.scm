; Python call extraction queries

; =============================================================================
; DIRECT FUNCTION CALLS
; =============================================================================

; Simple function call: func()
(call
  function: (identifier) @call.function_name
  arguments: (argument_list) @call.args) @call.direct

; =============================================================================
; METHOD CALLS
; =============================================================================

; Method call on object: obj.method()
(call
  function: (attribute
    object: (identifier) @call.receiver
    attribute: (identifier) @call.method_name)
  arguments: (argument_list) @call.method_args) @call.method

; Method call on self: self.method()
(call
  function: (attribute
    object: (identifier) @call.self_receiver
    attribute: (identifier) @call.self_method)
  arguments: (argument_list) @call.self_args
  (#eq? @call.self_receiver "self")) @call.self

; =============================================================================
; CHAINED CALLS
; =============================================================================

; Chained method call: obj.method1().method2()
(call
  function: (attribute
    object: (call) @call.chain_receiver
    attribute: (identifier) @call.chain_method)
  arguments: (argument_list) @call.chain_args) @call.chained

; =============================================================================
; NESTED ATTRIBUTE CALLS
; =============================================================================

; Call on nested attribute: a.b.method()
(call
  function: (attribute
    object: (attribute) @call.nested_receiver
    attribute: (identifier) @call.nested_method)
  arguments: (argument_list) @call.nested_args) @call.nested

; =============================================================================
; SPECIAL CALLS
; =============================================================================

; Super call: super()
(call
  function: (identifier) @call.super_name
  (#eq? @call.super_name "super")) @call.super

; Class instantiation (PascalCase identifier)
(call
  function: (identifier) @call.class_name
  arguments: (argument_list) @call.class_args
  (#match? @call.class_name "^[A-Z]")) @call.instantiation
