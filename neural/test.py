from lark import Lark

grammar = """
start: conv2d
conv2d: "Conv2D(" params ")"
params: param ("," param)* ","?
?param: number | tuple_ | STRING
number: NUMBER
tuple_: "(" number "," number ")"
STRING: "\\"" /[^"]+/ "\\""
NUMBER: /\\d+/
WS: /[ \\t]+/
%ignore WS
"""

parser = Lark(grammar, parser="lalr")
input_string = 'Conv2D(32, (3, 3), "relu")'
tree = parser.parse(input_string)
print(tree.pretty())
