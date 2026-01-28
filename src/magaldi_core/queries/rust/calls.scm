; Rust call extraction queries

; =============================================================================
; FUNCTION CALLS
; =============================================================================

; Simple function call: func()
(call_expression
  function: (identifier) @call.name
  arguments: (arguments) @call.args) @call.simple

; Scoped function call: module::func()
(call_expression
  function: (scoped_identifier) @call.scoped_name
  arguments: (arguments) @call.args) @call.scoped

; =============================================================================
; METHOD CALLS
; =============================================================================

; Method call: obj.method()
(call_expression
  function: (field_expression
    value: (_) @call.receiver
    field: (field_identifier) @call.method)
  arguments: (arguments) @call.args) @call.method

; Chained method call: obj.a().b()
(call_expression
  function: (field_expression
    value: (call_expression) @chain.parent
    field: (field_identifier) @chain.method)
  arguments: (arguments) @chain.args) @chain.call

; =============================================================================
; ASSOCIATED FUNCTION CALLS
; =============================================================================

; Type::new() or Type::from()
(call_expression
  function: (scoped_identifier
    path: (identifier) @associated.type
    name: (identifier) @associated.function)
  arguments: (arguments) @associated.args) @associated.call

; Complex path: module::Type::new()
(call_expression
  function: (scoped_identifier
    path: (scoped_identifier) @qualified_associated.path
    name: (identifier) @qualified_associated.function)
  arguments: (arguments) @qualified_associated.args) @qualified_associated.call

; =============================================================================
; MACRO CALLS
; =============================================================================

; println!(), vec![], format!()
(macro_invocation
  macro: (identifier) @macro_call.name
  (token_tree) @macro_call.args) @macro_call.call

; Scoped macro: log::info!()
(macro_invocation
  macro: (scoped_identifier) @macro_call.scoped_name
  (token_tree) @macro_call.args) @macro_call.scoped

; =============================================================================
; SPECIAL CALLS
; =============================================================================

; await expression: expr.await
(await_expression
  (_) @await.expression) @await.call

; Unwrap: .unwrap(), .expect()
(call_expression
  function: (field_expression
    field: (field_identifier) @unwrap.method)
  arguments: (arguments) @unwrap.args)
  (#match? @unwrap.method "^(unwrap|expect|unwrap_or|unwrap_or_else|unwrap_or_default)$") @unwrap.call

; Try operator: expr?
(try_expression
  (_) @try.expression) @try.call

; =============================================================================
; CLOSURE CALLS
; =============================================================================

; Closure call: closure()
(call_expression
  function: (parenthesized_expression
    (closure_expression) @closure_call.closure)
  arguments: (arguments) @closure_call.args) @closure_call.call

; Iterator methods: .map(), .filter(), etc.
(call_expression
  function: (field_expression
    field: (field_identifier) @iterator.method)
  arguments: (arguments
    (closure_expression) @iterator.closure))
  (#match? @iterator.method "^(map|filter|fold|for_each|find|any|all|flat_map|filter_map)$") @iterator.call
