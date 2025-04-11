from neural.parser.parser import network_parser

print("Testing lexer...")
tokens = list(network_parser.lex('network TestNet { input: (28, 28, 1) layers: Dense(128) ; }'))
print(f"Lexer works! Found {len(tokens)} tokens")

print("Done!")
