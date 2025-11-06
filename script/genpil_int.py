#!/usr/bin/env python3

def int_regs(bits, name, *args):
    yield from (f'$ int{bits}.{name}:{n} Int{bits}\n' for n in args)
def int_1i1o(bits, name):
    yield from int_regs(bits, name, 'x', 'result')
def int_2i1o(bits, name):
    yield from int_regs(bits, name, 'x', 'y', 'result')

def int_op(bits, op):
    yield """
> int*.%
. int*.to_32:value int*.%:x
| int*.to_32
. int32.%:x int*.to_32:result
. int*.to_32:value int*.%:y
| int*.to_32
. int32.%:y int*.to_32:result
| int32.%
. int32.to_*:value int32.%:result
| int32.to_*
. int*.%:result int32.to_*:result
<
""".replace('*', str(bits)).replace('%', op)
    yield from int_2i1o(bits, op)

def int_op_unary(op, bits):
    yield f"""
> int{bits}.{op}
. int{bits}.to_32:value int{bits}.{op}:x
| int{bits}.to_32
. int32.{op}:x int{bits}.to_32:result
| int32.{op}
. int32.to_{bits}:value int32.{op}:result
| int32.to_{bits}
. int{bits}.{op}:result int32.to_{bits}:result
<
"""
    yield from int_1i1o(bits, op)

def int_op_to(x, to):
    yield f"""
> int{x}.to_{to}
. int{x}.to_32:value int{x}.to_{to}:value
| int{x}.to_32
. int32.to_{to}:value int{x}.to_32:result
| int32.to_{to}
. int{x}.to_{to}:result int32.to_{to}:result
<
$ int{x}.to_{to}:value Int{x}
$ int{x}.to_{to}:result Int{to}
"""

def int_ops(bits):
    yield f"""
# builtin: int{bits}.to_32
$ int{bits}.to_32:value Int{bits}
$ int{bits}.to_32:result Int32

# builtin: int32.to_{bits}
$ int32.to_{bits}:value Int32
$ int32.to_{bits}:result Int{bits}
"""
    for op in ('nand', 'and', 'or', 'xor', 'sll', 'srl', 'add', 'sub'):
        yield from int_op(bits, op)
    yield from int_op_unary('inc', bits)
    yield from int_op_unary('dec', bits)
    yield from int_op_unary('not', bits)
    yield f"""
> int{bits}.eq
. int{bits}.to_32:value int{bits}.eq:x
| int{bits}.to_32
. int{bits}.eq@t int{bits}.to_32:result
. int{bits}.to_32:value int{bits}.eq:y
| int{bits}.to_32
. int32.eq:y int{bits}.to_32:result
. int32.eq:x int{bits}.eq@t
| int32.eq
. int{bits}.eq:result int32.eq:result
<
$ int{bits}.eq:x Int{bits}
$ int{bits}.eq:y Int{bits}
$ int{bits}.eq:result Int1
$ int{bits}.eq@t Int32
"""

    yield f"""
> *.is_between
. *.to_32:value *.is_between:x
| *.to_32
. int32.is_between:x *.to_32:result
. *.to_32:value *.is_between:min
| *.to_32
. int32.is_between:min *.to_32:result
. *.to_32:value *.is_between:max
| *.to_32
. int32.is_between:max *.to_32:result
| int32.is_between
. *.is_between:result int32.is_between:result
<
$ *.is_between:x Int{bits}
$ *.is_between:min Int{bits}
$ *.is_between:max Int{bits}
$ *.is_between:result Int1
""".replace('*', f'int{bits}')

def int32_cmp():
    for name, case, result in (('eq', 0, 1), ('ne', 0, 0), ('gt', 1, 1), ('lt', -1, 1), ('ge', -1, 0), ('le', 1, 0)):
        yield f"""
> int32.{name}
. int32.cmp:x int32.{name}:x
. int32.cmp:y int32.{name}:y
| int32.cmp
? int32.cmp:result
" {case} int32.{name}:{result}
= int32.{name}:{1 - result}

> int32.{name}:0
+ int32.{name}:result 0
<

> int32.{name}:1
+ int32.{name}:result 1
<

$ int32.{name}:x Int32
$ int32.{name}:y Int32
$ int32.{name}:result Int1
"""

def builtin(fn, regs):
    yield '\n'
    yield f'# builtin: {fn}\n'
    for reg, ty in regs:
        yield f'$ {fn}:{reg} {ty}\n'

def all_ops():
    yield open('_int.pil').read()
    yield from int32_cmp()
    for i in range(1, 32):
        yield from int_ops(i)
        for k in range(1, 32):
            if i != k:
                yield from int_op_to(i, k)

if __name__ == '__main__':
    print(*all_ops(), sep='', end='')
