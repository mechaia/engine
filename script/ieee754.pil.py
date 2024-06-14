#!/usr/bin/env python3

def op_cmp(name):
    yield f"""\
> fp32.{name}
| fp32.sub
| fp32.sign
= @{name}
"""

def op_other():
    yield """\
> fp32.neg
. fp32:1 fp32:0
= fp32.sub
"""

def all_ops():
    yield from op_cmp('lt')
    yield from op_cmp('eq')
    yield from op_cmp('gt')
    yield from op_other()
    for i in range(2, 8):
        yield f'$ fp32:{i} Fp32'

if __name__ == '__main__':
    print(*all_ops(), sep='\n', end='')
