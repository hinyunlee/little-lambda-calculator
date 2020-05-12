# Little λ Calculator
A very basic calculator that does math and lambda calculus.

# TODO

## Features
* Numbers: Integer, Rational
* Logic: Boolean
* Operators
* Lazy evaluation
* Memoization
* Lambda expression: "λ"
* Conditional expression: "if"
* Lexical scoping: "let"
* Imperative programming: "begin"

## Requirements
* Python 3

## Run
### REPL
```
python run.py
```

### Load file
```
python run.py <filename>
```

### Run tests
```
python test.py
```

## Guide
```
<Var>: Variable
<Exp>: Expression
```

### Logic
```
true
false
¬ <Exp>
<Exp> ∧ <Exp>
<Exp> ∨ <Exp>
```

### Mathematics
```
(<Exp>)
<Exp> ↑ <Exp>
- <Exp>
<Exp> + <Exp>
<Exp> - <Exp>
<Exp> ⋅ <Exp>
<Exp> / <Exp>
...
```

### Lambda calculus
```
λ<Var>.<Exp>
```
```
<Exp> <Exp>
```

### Let expression
```
let <Var> := <Exp> in <Exp>
```
```
let <Var> := <Exp>;<Var> := <Exp>;...;<Var> := <Exp> in <Exp>
```
```
let
    <Var> := <Exp>
    <Var> := <Exp>
    ...
    <Var> := <Exp>
in
    <Exp>
```

### Recursive let expression
```
let rec <Var> := <Exp> in <Exp>
```
```
let rec <Var> := <Exp>;<Var> := <Exp>;...;<Var> := <Exp> in <Exp>
```
```
let rec
    <Var> := <Exp>
    <Var> := <Exp>
    ...
    <Var> := <Exp>
in
    <Exp>
```

### Conditional expression
```
if <Exp> then <Exp> else <Exp>
```
```
<Exp> if <Exp> else <Exp>
```

### Imperative programming
```
begin <Exp> end
```
```
begin <Exp>;<Exp>;...;<Exp> end
```
```
begin
    <Exp>
    <Exp>
    ...
    <Exp>
end
```

### Definition
```
<Var> := <Exp>
```
```
<Var> := <Exp>;<Var> := <Exp>;...;<Var> := <Exp>
```
```
<Var> := <Exp>
<Var> := <Exp>
...
<Var> := <Exp>
```

### Operators
```
(Highest precedence to lowest precedence)
()
↑ (right-associative)
¬, - (prefix)
⋅, /, ∧
+, -, ∨
<, ≤, >, ≥, =, ≠
∘ (right-associative)
λ
if, let, begin
```

## Examples
See ./examples