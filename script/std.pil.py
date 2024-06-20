#!/usr/bin/env python3

def int_op_add(bits):
    yield """
> int*.add
| int*.to_32
. int32:1 int32:0
. int*:0 int*:1
| int*.to_32
| int32.add
= int32.to_*
""".replace('*', str(bits))

def int_op_unary(op, bits):
    yield f"""
> int{bits}.{op}
| int{bits}.to_32
| int32.{op}
= int32.to_{bits}
"""

def int_op_inc32():
    yield """
> int32.inc
+ int32:1 -1
= int32.sub
"""

def int_op_dec32():
    yield """
> int32.dec
+ int32:1 1
= int32.sub
"""

def int_op_add32():
    yield """
> int32.add
. @Int32:0 int32:0
+ int32:0 0
| int32.sub
. int32:1 int32:0
. int32:0 @Int32:0
= int32.sub
"""

def int_op_neg32():
    yield """
> int32.neg
. int32:1 int32:0
+ int32:0 0
= int32.sub
"""

def int_op_to(from_bits, to_bits):
    yield f"""
> int{from_bits}.to_{to_bits}
| int{from_bits}.to_32
= int32.to_{to_bits}
"""

def int_ops(bits):
    yield f"$ int{bits}:0 Int{bits}\n"
    yield f"$ int{bits}:1 Int{bits}\n"
    yield from int_op_add(bits)
    yield from int_op_unary('inc', bits)
    yield from int_op_unary('dec', bits)

def write_ops():
    yield """
> write.newline
+ int8:0 '\\n'
= write.byte

> write.const_str
+ int32:0 0
= write.const_str:cond
> write.const_str:loop
. int32:0 @Int32:0
| const_str.get
| write.byte
| int32.inc
= write.const_str:cond
> write.const_str:cond
. @Int32:0 int32:0
. int32:1 int32:0
| const_str.len
| int32.gt
= write.const_str:switch
[ write.const_str:switch int1:0
? 1 write.const_str:loop
! noop

> write.int32
. @Int32:1 int32:0
+ int32:1 0
| int32.lt
= write.int32:switch
[ write.int32:switch int1:0
? 0 write.nat32:loop
! write.int32:negative
> write.int32:negative
+ int8:0 '-'
| write.byte
. int32:0 @Int32:1
| int32.neg
. @Int32:1 int32:0
= write.nat32:loop

> write.nat32
. @Int32:1 int32:0
= write.nat32:loop
> write.nat32:loop
. int32:0 @Int32:1
+ int32:1 10
| int32.divmod
. @Int32:1 int32:0
+ int32:0 '0'
| int32.add
| int32.to_8
| @write_stack.push
= write.nat32:switch
[ write.nat32:switch @Int32:1
? 0 @write_stack
! write.nat32:loop

> @write_stack.push
{ @Int8 @Int8:index int8:0
. int5:0 @Int8:index
| int5.inc
. @Int8:index int5:0
<

> @write_stack
= @write_stack:switch
> @write_stack:loop
. int5:0 @Int8:index
| int5.dec
. @Int8:index int5:0
} @Int8 @Int8:index int8:0
| write.byte
= @write_stack:switch
[ @write_stack:switch @Int8:index
? 0 noop
! @write_stack:loop
"""

def other_ops():
    yield """\
% IntSign 0 1 -1

$ @Int32:0 Int32
$ @Int32:1 Int32
$ @Int8:index Int5
@ @Int8 Int5 Int8

> noop
<

> @true
+ int1:0 1
<

> @false
+ int1:0 0
<


[ @lt intsign:0
? -1 @true
! @false

[ @eq intsign:0
? 0 @true
! @false

[ @gt intsign:0
? 1 @true
! @false


[ @ge intsign:0
? -1 @false
! @true

[ @ne intsign:0
? 0 @false
! @true

[ @le intsign:0
? 1 @false
! @true
"""

def int_op_32():
    yield from int_op_add32()
    yield from int_op_neg32()
    yield from int_op_inc32()
    yield from int_op_dec32()
    yield """\
> int32.cmp
| int32.sub
= int32.sign


> int32.bitnot
. int32:1 int32:0
= int32.bitnand

> int32.bitand
| int32.bitnand
= int32.bitnot

> int32.bitor
. @Int32:0 int32:1
| int32.bitnot
. @Int32:1 int32:0
. int32:0 @Int32:0
| int32.bitnot
. int32:1 @Int32:1
= int32.bitnand

> int32.bitxor
. @Int32:1 int32:1
. @Int32:0 int32:0
| int32.bitor
. int32:1 @Int32:1
. @Int32:1 int32:0
. int32:0 @Int32:0
| int32.bitnand
. int32:1 @Int32:1
= int32.bitand

"""
    for name in ('eq', 'ne', 'gt', 'lt', 'ge', 'le'):
        yield f"""\
> int32.{name}
| int32.cmp
= @{name}
"""

def all_ops():
    yield from other_ops()
    yield from write_ops()
    yield from int_op_32()
    for i in range(1, 32):
        yield from int_ops(i)
    for i in range(1, 32):
        for k in range(1, 32):
            if i != k:
                yield from int_op_to(i, k)

    yield '$ int32:0 Int32\n'
    yield '$ int32:1 Int32\n'
    yield '$ intsign:0 IntSign\n'

if __name__ == '__main__':
    print(*all_ops(), sep='', end='')
