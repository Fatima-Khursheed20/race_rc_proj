import sys
print(f"Python {sys.version}")

try:
    with open('src/distractor_generator.py', 'r') as f:
        code = f.read()
    compile(code, 'src/distractor_generator.py', 'exec')
    print("distractor_generator.py: OK")
except SyntaxError as e:
    print(f"distractor_generator.py: SYNTAX ERROR at line {e.lineno}")
    print(f"  {e.msg}")
except Exception as e:
    print(f"distractor_generator.py: ERROR - {e}")

try:
    with open('src/model_b_train.py', 'r') as f:
        code = f.read()
    compile(code, 'src/model_b_train.py', 'exec')
    print("model_b_train.py: OK")
except SyntaxError as e:
    print(f"model_b_train.py: SYNTAX ERROR at line {e.lineno}")
    print(f"  {e.msg}")
except Exception as e:
    print(f"model_b_train.py: ERROR - {e}")
