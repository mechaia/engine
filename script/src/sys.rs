use crate::{BuiltinType, Collection, Error, ErrorKind, Function, Type};

const INT_32_TO_X: u32 = 0 << 5;
const NAT_X_TO_32: u32 = 1 << 5;
const INT_X_TO_32: u32 = 2 << 5;

const INT_32_SUB: u32 = 3 << 5 | 0;
const INT_32_BITNAND: u32 = 3 << 5 | 1;
const INT_32_SIGN: u32 = 3 << 5 | 3;
const NAT_32_DIVMOD: u32 = 3 << 5 | 4;

const INT_SIGN_TO_2: u32 = 3 << 5 | 6;
const INT_2_TO_SIGN: u32 = 3 << 5 | 7;

const CONST_STR_GET: u32 = 3 << 5 | 8;
const CONST_STR_LEN: u32 = 3 << 5 | 9;

const WRITE_BYTE: u32 = 3 << 5 | 15;

const FP_32_ADD: u32 = 4 << 5 | 0;
const FP_32_SUB: u32 = 4 << 5 | 1;
const FP_32_MUL: u32 = 4 << 5 | 2;
const FP_32_DIV: u32 = 4 << 5 | 3;
const FP_32_SIGN: u32 = 4 << 5 | 4;
const FP_32_SQRT: u32 = 4 << 5 | 5;
const FP_32_SIGN_INT: u32 = 4 << 5 | 29;
const FP_32_TO_INT_BITS: u32 = 4 << 5 | 30;
const FP_32_FROM_INT_BITS: u32 = 4 << 5 | 31;

pub(crate) fn add_defaults(collection: &mut Collection) -> Result<(), Error> {
    add_integer(collection)?;
    add_const_str(collection)?;
    add_write(collection)?;
    Ok(())
}

pub(crate) fn add_ieee754(c: &mut Collection) -> Result<(), Error> {
    c.types
        .try_insert("Fp32".into(), Type::Builtin(BuiltinType::Fp32))
        .unwrap();

    let mut f = |n, s| add_fn(c, n, s, ["fp32:0", "fp32:1"], ["fp32:0"]);
    f("fp32.add", FP_32_ADD)?;
    f("fp32.sub", FP_32_SUB)?;
    f("fp32.mul", FP_32_MUL)?;
    f("fp32.div", FP_32_DIV)?;
    let mut f = |n, s| add_fn(c, n, s, ["fp32:0"], ["fp32:0"]);
    f("fp32.sign", FP_32_SIGN)?;
    f("fp32.sqrt", FP_32_SQRT)?;

    add_fn(
        c,
        "fp32.sign.int",
        FP_32_SIGN_INT,
        ["fp32:0"],
        ["intsign:0"],
    )?;
    add_fn(
        c,
        "fp32.to_int_bits",
        FP_32_TO_INT_BITS,
        ["fp32:0"],
        ["int32:0"],
    )?;
    add_fn(
        c,
        "fp32.from_int_bits",
        FP_32_FROM_INT_BITS,
        ["int32:0"],
        ["fp32:0"],
    )?;

    Ok(())
}

fn add_integer(c: &mut Collection) -> Result<(), Error> {
    for i in 1..=31 {
        let ty = format!("Int{i}").into_boxed_str();
        let arg = format!("int{i}:0").into_boxed_str();

        let mut f = |f, s, x, y| add_fn(c, f, s | u32::from(i), [x], [y]);
        f(format!("int32.to_{i}"), INT_32_TO_X, "int32:0", &*arg)?;
        f(format!("int{i}.to_32"), INT_X_TO_32, &*arg, "int32:0")?;
        f(format!("nat{i}.to_32"), NAT_X_TO_32, &*arg, "int32:0")?;

        c.types
            .try_insert(ty, Type::Builtin(BuiltinType::Int(i)))
            .unwrap();
    }

    c.types
        .try_insert("IntSign".into(), Type::Builtin(BuiltinType::IntSign))
        .unwrap();
    add_fn(c, "intsign.to_2", INT_SIGN_TO_2, ["intsign:0"], ["int2:0"])?;
    add_fn(c, "int2.to_sign", INT_2_TO_SIGN, ["int2:0"], ["intsign:0"])?;

    c.types
        .try_insert("Int32".into(), Type::Builtin(BuiltinType::Int(32)))
        .unwrap();

    add_fn(
        c,
        "int32.sub",
        INT_32_SUB,
        ["int32:0", "int32:1"],
        ["int32:0"],
    )?;
    add_fn(c, "int32.sign", INT_32_SIGN, ["int32:0"], ["intsign:0"])?;
    add_fn(
        c,
        "int32.divmod",
        NAT_32_DIVMOD,
        ["int32:0", "int32:1"],
        ["int32:0", "int32:1"],
    )?;

    add_fn(c, "int32.bitnand", INT_32_BITNAND, ["int32:0", "int32:1"], ["int32:0"])?;

    Ok(())
}

fn add_const_str(c: &mut Collection) -> Result<(), Error> {
    c.types
        .try_insert(
            "ConstStr".into(),
            Type::Builtin(BuiltinType::ConstantString),
        )
        .unwrap();
    c.registers
        .try_insert("const_str:0".into(), "ConstStr".into())
        .unwrap();
    add_fn(
        c,
        "const_str.get",
        CONST_STR_GET,
        ["const_str:0", "int32:0"],
        ["int8:0"],
    )?;
    add_fn(
        c,
        "const_str.len",
        CONST_STR_LEN,
        ["const_str:0"],
        ["int32:0"],
    )?;
    Ok(())
}

fn add_write(c: &mut Collection) -> Result<(), Error> {
    add_fn(c, "write.byte", WRITE_BYTE, ["int8:0"], [])
}

fn add_fn<S: Into<Box<str>>>(
    collection: &mut Collection,
    name: impl Into<Box<str>>,
    id: u32,
    inputs: impl IntoIterator<Item = S>,
    outputs: impl IntoIterator<Item = S>,
) -> Result<(), Error> {
    _add_fn(
        collection,
        name.into(),
        id,
        inputs.into_iter().map(Into::into).collect(),
        outputs.into_iter().map(Into::into).collect(),
    )
}

fn _add_fn(
    collection: &mut Collection,
    name: Box<str>,
    id: u32,
    inputs: Box<[Box<str>]>,
    outputs: Box<[Box<str>]>,
) -> Result<(), Error> {
    collection
        .functions
        .try_insert(
            name,
            Function::Builtin {
                id,
                inputs,
                outputs,
            },
        )
        .map(|_| ())
        .map_err(|e| Error {
            line: 0,
            kind: ErrorKind::DuplicateFunction(e.entry.key().clone().into()),
        })
}

pub mod wordvm {
    use {
        super::*,
        crate::executor::{
            wordvm::{WordVM, WordVMState},
            Error,
        },
    };

    pub enum External {
        WriteByte(u8),
    }

    struct Handler<'a> {
        input: u32,
        output: u32,
        vm: &'a WordVM,
        state: &'a mut WordVMState,
    }

    impl<'a> Handler<'a> {
        fn input(&self, i: u32) -> Result<u32, Error> {
            self.state.register(self.vm.constant(self.input + i)?)
        }

        fn output(&mut self, i: u32, value: u32) -> Result<(), Error> {
            self.state
                .set_register(self.vm.constant(self.output + i)?, value)
        }

        fn in1out1<F: FnOnce(u32) -> u32>(&mut self, f: F) -> Result<(), Error> {
            self.output(0, f(self.input(0)?))
        }

        fn in2out1<F: FnOnce(u32, u32) -> u32>(&mut self, f: F) -> Result<(), Error> {
            self.output(0, f(self.input(0)?, self.input(1)?))
        }

        fn in1out1_fp<F: FnOnce(f32) -> f32>(&mut self, f: F) -> Result<(), Error> {
            self.in1out1(|x| f(f32::from_bits(x)).to_bits())
        }

        fn in2out1_fp<F: FnOnce(f32, f32) -> f32>(&mut self, f: F) -> Result<(), Error> {
            self.in2out1(|x, y| f(f32::from_bits(x), f32::from_bits(y)).to_bits())
        }
    }

    pub fn handle(
        vm: &WordVM,
        state: &mut WordVMState,
        id: u32,
    ) -> Result<Option<External>, Error> {
        let (input, output) = vm.sys_registers(id);
        let mut h = Handler {
            input,
            output,
            vm,
            state,
        };

        match id {
            CONST_STR_GET => {
                let offt = h.input(0)?;
                let len = h.input(1)?;
                let index = h.input(2)?;
                let c = vm.strings_buffer()[offt as usize..][..len as usize]
                    .get(index as usize)
                    .copied()
                    .unwrap_or(u8::MAX);
                h.output(0, u32::from(c))?;
            }
            CONST_STR_LEN => h.in2out1(|_, y| y)?,
            INT_32_SUB => h.in2out1(|x, y| x.wrapping_sub(y))?,
            INT_32_BITNAND => h.in2out1(|x, y| !(x & y))?,
            INT_32_SIGN => h.in1out1(int_signum)?,
            NAT_32_DIVMOD => {
                let (x, y) = (h.input(0)?, h.input(1)?);
                h.output(0, x / y)?;
                h.output(1, x % y)?;
            }
            WRITE_BYTE => return Ok(Some(External::WriteByte(h.input(0)? as u8))),

            // 0 => 0, 1 => 1, 2 => 3 (-1)
            INT_SIGN_TO_2 => h.in1out1(|x| x | (x >> 1))?,
            INT_2_TO_SIGN => h.in1out1(int2_intsign)?,

            FP_32_ADD => h.in2out1_fp(|x, y| x + y)?,
            FP_32_SUB => h.in2out1_fp(|x, y| x - y)?,
            FP_32_MUL => h.in2out1_fp(|x, y| x * y)?,
            FP_32_DIV => h.in2out1_fp(|x, y| x / y)?,
            FP_32_SIGN => h.in1out1_fp(super::f32_signum)?,
            FP_32_SQRT => h.in1out1_fp(f32::sqrt)?,
            FP_32_SIGN_INT => h.in1out1(|x| {
                let s = super::f32_signum(f32::from_bits(x));
                int2_intsign(s as i32 as u32 & 3)
            })?,
            FP_32_TO_INT_BITS | FP_32_FROM_INT_BITS => h.in1out1(|x| x)?,

            id => match id & !0x1f {
                // make sure to mask so array indices work as expected
                INT_32_TO_X => {
                    let bits = id & 0x1f;
                    let mask = (1 << bits) - 1;
                    h.in1out1(|x| x & mask)?;
                }
                NAT_X_TO_32 => {
                    let shift = 32 - (id & 0x1f);
                    h.in1out1(|x| (x << shift) >> shift)?;
                }
                INT_X_TO_32 => {
                    let shift = 32 - (id & 0x1f);
                    h.in1out1(|x| ((x << shift) as i32 >> shift) as u32)?;
                }
                _ => Err(Error)?,
            },
        }
        Ok(None)
    }

    fn int_signum(n: u32) -> u32 {
        int2_intsign((n as i32).signum() as u32 & 3)
    }

    /// Convert int2 to a intsign, which is uses bitpatterns 0b00, 0b01 and 0b10 for 0, 1 and -1 respectively
    fn int2_intsign(x: u32) -> u32 {
        x & !(x >> 1)
    }
}

/// Signum for IEEE 754 floating point numbers.
///
/// Unlike [`core::f32::signum`], this version does account for zero numbers.
///
/// The exact mapping is:
/// - `NaN` -> `NaN` (bits preserved)
/// - `x` > +0.0 -> +1.0
/// - `x` < -0.0 -> -1.0
/// - `x` == +0.0 -> +0.0
/// - `x` == -0.0 -> -0.0
fn f32_signum(x: f32) -> f32 {
    if x != x {
        return x;
    }
    let mut n = x.to_bits();
    if (n & 0x7fffffff) != 0 {
        n |= 0x3f800000;
    }
    n &= 0xbf800000;
    f32::from_bits(n)
}
