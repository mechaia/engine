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

def int_op_inc32():
    yield """
> int32.inc
+ int32:1 -1
= int32.sub
"""

def int_op_add32():
    yield """
> int32.add
. @32 int32:0
+ int32:0 0
| int32.sub
. int32:1 int32:0
. int32:0 @32
= int32.sub
"""

def int_op_neg32():
    yield """
> int32.neg
. int32:1 int32:0
+ int32:0 0
= int32.sub
"""

def int_op_cmp32(name, cond):
    yield f"""
> int32.{name}
| int32.sub
| int32.sign
= int32.{name}:switch
[ int32.{name}:switch int2:0
? {cond} @true
! @false
"""

def int_op_lt32():
    yield from int_op_cmp32('lt', '-1')

def int_op_gt32():
    yield from int_op_cmp32('gt', '1')

def int_ops(bits):
    yield f"$ int{bits}:1 Int{bits}\n"
    yield from int_op_add(bits)

def write_ops():
    yield """
> write.newline
+ int8:0 '\\n'
= write.byte

> write.const_str
+ int32:0 0
= write.const_str:cond
> write.const_str:loop
. int32:0 @32
| const_str.get
| write.byte
| int32.inc
= write.const_str:cond
> write.const_str:cond
. @32 int32:0
. int32:1 int32:0
| const_str.len
| int32.gt
= write.const_str:switch
[ write.const_str:switch int1:0
? 1 write.const_str:loop
! @noop

> write.int32
. @32 int32:0
+ int32:1 0
| int32.lt
= write.int32:switch
[ write.int32:switch int1:0
? 0 write.nat32:loop
! write.int32:negative
> write.int32:negative
+ int8:0 '-'
| write.byte
. int32:0 @32
| int32.neg
. @32 int32:0
= write.nat32:loop

> write.nat32
. @32 int32:0
= write.nat32:loop
> write.nat32:loop
. int32:0 @32
+ int32:1 10
| int32.divmod
. @32 int32:0
+ int32:0 '0'
| int32.add
. int32:0 int32:1
| write.byte
= write.nat32:switch
[ write.nat32:switch @32
? 0 @noop
! write.nat32:loop
"""

def other_ops():
    yield """\
$ @32 Int32

> @noop
<

> @true
+ int1:0 1
<

> @false
+ int1:0 0
<
"""

def all_ops():
    yield from other_ops()
    yield from write_ops()
    for i in range(1, 32):
        yield from int_ops(i)
    yield from int_op_add32()
    yield from int_op_neg32()
    yield from int_op_inc32()
    yield from int_op_lt32()
    yield from int_op_gt32()

if __name__ == '__main__':
    print(*all_ops(), sep='', end='')
