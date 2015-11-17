#Ken (Quan) Zhou
#Assignment #2
#1.(a)
import re
def variable(tmp, top = True):
    tokens = tmp[0:]
    if re.match(r'([a-z]{1}[a-z][A-Z][0-9]+)', tokens[0]):
        r = (int(tokens[0]), tokens[1:])
        if not top or len(tokens) == 0:
            return r

def number(tmp, top = True):
    tokens = tmp[0:]
    if re.match(r'0|[+]?\d+', tokens[0]):
        r = (str(tokens[0]), tokens[1:])      
        if not top or len(tokens) == 0:
            return r
        
#1.(b)
def formula(tmp, top = True): #define function for "formula"
    tokens = tmp[0:]
    r = leftformula(tokens, False)
    if not r is None:
        (e1, tokens) = r
        if tokens[0] == 'and':
            r = formula(tokens[1:])
            if not r is None:
                (e2, tokens)  = r
                if not top or len(tokens) == 0:
                    return ({'And': [e1,e2]}, tokens[1:])
    
def leftformula(tkns, top = True): #defining a left recursive function
    tokens = tkns[0:]
    if tokens[0] == 'true':
        tokens = tokens[1:]
        if not top or len(tokens) == 0:
            return ('True', tokens[1:])
    elif tokens[0] == 'false':
        tokens = tokens[1:]
        if not top or len(tokens) == 0:
            return ('False', tokens[1:])
    elif tokens[0] == 'not' and tokens[1] == '(': 
        r = formula(tokens[2:], False)
        if not r is None:
            (e, tokens) = r
            if tokens[0] == ')':
                tokens = tokens[1:]
                if not top or len(tokens) == 0:
                    return ({'Not': [e]}, tokens[1:])
    elif tokens[0] == 'nonzero' and tokens[1] == '(':
        r = term(tokens[2:], False)
        if not r is None:
            (e, tokens) = r
            if tokens[0] == ')':
                tokens = tokens[1:]
                if not top or len(tokens) == 0:
                    return ({'Nonzero': [e]}, tokens[1:])
    else:
        r = variable(tokens, False)
#1.(c)
def term(tmp, top = True): #define function for "formula"
    tokens = tmp[0:]
    r = factor(tokens, False)
    if not r is None:
        (e, tokens) = r
        if tokens[0] == '+':
            r = term(tokens[1:], False)
            if not top or len(tokens) == 0: 
                return ({'Plus': [e]}, tokens[1:])      

def factor(tokns, top = True): #define function for "factor"
    tokens = tokn[0:]
    r = leftfactor(tokens, False)
    if not r is None:
        (e, tokens) = r
        if tokens[0] == '*':
            r = factor(tokens[1:], False)
            if not top or len(tokens) == 0:
                return ({'Mult': [e]}, tokens[1:])
def leftfactor(toks, top = True): #defining a left recursive function
    tokens = toks[0:]
    if token[0] == '(':
        r = term(tokens[1:])
        if not r is None:
            (e1, tokens) = r
            if tokens[0] == ')':
                tokens = tokens[1:]
                if not top or len(tokens) == 0:
                    return ({'Parens': [e]}, tokens[1:])
    elif tokens[0] == 'if' and tokens[1] == '(':
        r = formula(tokens[2:], False)
        if not r is None:
            (e1, tokens) = r
            if tokens[0] == ',':
                r = term(tokens[1:], False)
                if not r is None:
                    (e2, tokens) = r
                    if tokens[0] ==',':
                        r = term(tokens[1:], False)
                        if not r is None:
                             (e3, tokens) = r
                             if tokens[0] == ')':
                                 tokens = tokens[1:]
                                 if not top or len(tokens) == 0:
                                     return ({'If': [e1, e2, e3]}, tokens)
    elif type(tokens[0]) == int:
        return ('Number', tokens[1:])
    else:
        r = variable(tokens)
#1 (d)
def expression(tmp, top = True): #define helper parser function
    tokens = tmp[0:]
    () = term(tokens, False)
    () = formula(tokens, False)
    
def program(tmp, top = True): #define function of "program"
    tokens = tmp[0:]
    if tokens[0] == 'print':
        r = expression(tokens[1:], False) #call function "term"
        if not r is None:
            (e1, tokens) = r
            if tokens[0] == ';':
                r = program(tokens[1:], False) #recursive functionc call
                if not r is None:
                    (e2, tokens) = r
                    if not top or len(tokens) == 0:
                        return ({'Print': [e1, e2]}, tokens)
    elif tokens[0] == 'assign':
        r = variable(tokens[1:], False) #call function of variable
        if not r is None:
            (e1, tokens) = r
            if tokens[0] == ':' and tokens[1] == '=':
                r = expression(tokens[2:], False) #call function of "term"
                if not r is None:
                    (e2, tokens) = r
                    if tokens[0] ==';':
                        r = program(tokens[1:], False) #recursive function call
                        if not r is None:
                            (e3, tokens) = r
                            if not top or len(tokens) == 0:
                                return ({'Assign': [e1, e2, e3]}, tokens)
    elif tokens[0] == 'if':
        r = expression(tokens[1:], False) #call function of variable
        if not r is None:
            (e1, tokens) = r
            if tokens[0] == '{':
                r = program(tokens[1:], False) #recursive functionc call
                if not r is None:
                    (e2, tokens) = r
                    if tokens[0] == '}':
                        r = program(tokens[1:], False)
                        if not r is none:
                            (e3, tokens) = r
                            if not top or len(tokens) == 0:
                                return ({'If': [e1, e2, e3]}, tokens)
    elif tokens[0] == 'do' and tokens[1] == '{':
        r = program(tokens[2:], False) #call function of variable
        if not r is None:
            (e1, tokens) = r
            if tokens[0] == '}' and tokens[1] == 'until':
                r = expression(tokens[2:], False) #call function of "term"
                if not r is None:
                    (e2, tokens) = r
                    if tokens[0] ==';':
                        r = program(tokens[1:], False) #recursive function call
                        if not r is None:
                            (e3, tokens) = r
                            if not top or len(tokens) == 0:
                                return ({'DoUntil': [e1, e2, e3]}, tokens)
    elif len(tokens) == 0
        return ('End')
#Ken (Quan) Zhou
#Assignment #2
#2.(a)
Node = dict
Leaf = str

def evalTerm(env, t):
    if type(t) == Node:
        for label in t:
            children = t[label]
            if label == 'Mult':
                t1 = children[0]
                v1 = evalTerm(env, t1)
                t2 = children[1]
                v2 = evalTerm(env, t2)
                return v1*v2
            elif label == 'Plus':
                t1 = children[0]
                v1 = evalTerm(env, t1)
                t2 = children[1]
                v2 = evalTerm(env, t2)
                return v1+v2
            elif label == 'If':
                t = children[0]
                f1 = children[1]
                f2 = children[2]
                s = evalFormula(env, t)
                if s = True:
                    v1 = evalTerm(env, f1)
                    return v1
                else:
                    v2 = evalTerm(env, f2)
                    return v2
            elif label == 'Parens':
                t = children[0]
                v = evalTerm(env, t)
                return v
    elif type(t) == Leaf:
        for label in t:
            children = t[label]
            if label == 'Variable':
                x = children[0]
                if x in env:
                    return env[x]
                else:
                    print(x + "is unbound.")
                    exit()
            elif label == 'Number':
                n = children[0]
                return n            
#2. (b)
def evalFormula(env, f):
    if type(f) == Node:
        for label in f:
            children = f[label]
            if label == 'And':
                f1 = children[0]
                v1 = evalFormula(env, f1)
                f2 = children[1]
                v2 = evalFormula(env, f2)
                return v1 and v2 #python operator
            elif label == 'Not':
                f = children[0]
                v = evalFormula(env, t)
                return not v #python operator
            elif label == 'Nonzero':
                t = children[0]
                v = evalTerm(env, t)
                if v!= 0:
                    return True
                else:
                    return False
    elif type(f) == Leaf:
        for label in f:
            children = f[label]
            if label == 'Variable':
                x = children[0]
                if x in env:
                    return env[x]
                else:
                    print(x + "is unbound.")
                    exit()        
#2. (c)    
def evalexpression(env, s):
    
def execProgram(env, s):
    if type(s) == Leaf:
        if s == 'End':
            return []
    elif type(s) == Node:
        for label in s:
            if s == 'Print':
                children = s[label]
                e = children[0]
                p = children[1] 
                v = evalTerm(env, e) or v = evalFormula(env, e)
                (env, o) = execute(env, p)
                return (env, [v] + o)
            if s == 'Assign':
                children = s[label]
                x = children[0]['Variable'][0]
                e = children[1]
                p = children[2]
                v = evalTerm(env, e) or v = evalFormula(env, e)
                env[x] = v
                (env, o) = execute(env, p)
                return o
            if s == 'If':
                children = s[label]
                e = children[0]
                body = children[1]
                rest = children[2]
                v = evalTerm(env, e) or  v = evalFormula(env, e)
                if v = True:
                    (env2, o1) = execProgram(env1, body)
                    (env3, o2) = execProgram(env2, rest)
                    return (env3, o1, o2)
                else:
                    (env2, o1) = execProgram(env1, rest)
                    return (env2, o1)
            if s == 'While':
                children = s[label]
                e = children[0]
                body = children[1]
                rest = children[2]
                (env1, o1) = evalTerm(env, e) or  (env1, o1) = evalFormula(env, e)
                
                if v = True:
                    (env2, o1) = execProgram(env1, body)
                    (env3, o2) = execProgram(env2, rest)
                    return (env3, o1, o2)
                else:
                    (env2, o1) = execProgram(env1, rest)
                    return (env2, o1)
