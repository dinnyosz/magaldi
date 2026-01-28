; JavaScript/TypeScript call extraction queries

; =============================================================================
; FUNCTION CALLS
; =============================================================================

; Simple function call: func()
(call_expression
  function: (identifier) @call.name
  arguments: (arguments) @call.args) @call.simple

; Method call: obj.method()
(call_expression
  function: (member_expression
    object: (_) @call.object
    property: (property_identifier) @call.method)
  arguments: (arguments) @call.args) @call.method

; Chained method call: obj.a().b()
(call_expression
  function: (member_expression
    object: (call_expression) @call.chain_parent
    property: (property_identifier) @call.chained_method)
  arguments: (arguments) @call.args) @call.chained

; Optional chaining: obj?.method()
(call_expression
  function: (member_expression
    object: (_) @call.optional_object
    "?."
    property: (property_identifier) @call.optional_method)
  arguments: (arguments) @call.args) @call.optional

; =============================================================================
; CONSTRUCTOR CALLS
; =============================================================================

; new Constructor()
(new_expression
  constructor: (identifier) @new.name
  arguments: (arguments)? @new.args) @new.call

; new module.Constructor()
(new_expression
  constructor: (member_expression
    object: (_) @new.module
    property: (property_identifier) @new.name)
  arguments: (arguments)? @new.args) @new.qualified

; =============================================================================
; SPECIAL CALLS
; =============================================================================

; await call
(await_expression
  (call_expression
    function: (_) @await.function
    arguments: (arguments) @await.args)) @await.call

; IIFE (Immediately Invoked Function Expression)
(call_expression
  function: (parenthesized_expression
    (function_expression) @iife.function)
  arguments: (arguments) @iife.args) @iife.call

; Arrow IIFE
(call_expression
  function: (parenthesized_expression
    (arrow_function) @arrow_iife.function)
  arguments: (arguments) @arrow_iife.args) @arrow_iife.call

; =============================================================================
; CALLBACK PATTERNS
; =============================================================================

; Array methods with callbacks: arr.map(fn)
(call_expression
  function: (member_expression
    property: (property_identifier) @array_method.name)
  arguments: (arguments
    [
      (arrow_function) @array_method.arrow_callback
      (function_expression) @array_method.func_callback
      (identifier) @array_method.ref_callback
    ]))
  (#match? @array_method.name "^(map|filter|reduce|forEach|find|some|every|flatMap)$") @array_method.call

; Promise then/catch
(call_expression
  function: (member_expression
    property: (property_identifier) @promise.method)
  arguments: (arguments
    (_) @promise.handler))
  (#match? @promise.method "^(then|catch|finally)$") @promise.call
