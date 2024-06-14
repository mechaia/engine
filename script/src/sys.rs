use crate::{BuiltinType, Collection, Error, ErrorKind, Function, Type};

const INT_32_TO_X: u32 = 0 << 5;
const NAT_X_TO_32: u32 = 1 << 5;
const INT_X_TO_32: u32 = 2 << 5;

const INT_32_SUB: u32 = 3 << 5 | 0;
const INT_32_SIGN: u32 = 3 << 5 | 1;
const NAT_32_DIVMOD: u32 = 3 << 5 | 2;

const CONST_STR_GET: u32 = 3 << 5 | 8;
const CONST_STR_LEN: u32 = 3 << 5 | 9;

const WRITE_BYTE: u32 = 3 << 5 | 15;

const FP_32_ADD: u32 = 4 << 5 | 0;
const FP_32_SUB: u32 = 4 << 5 | 1;
const FP_32_MUL: u32 = 4 << 5 | 2;
const FP_32_DIV: u32 = 4 << 5 | 3;
const FP_32_SIGN: u32 = 4 << 5 | 4;
const FP_32_SQRT: u32 = 4 << 5 | 5;

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
    for a in 0..2 {
        c.registers
            .try_insert(format!("fp32:{a}").into(), "Fp32".into())
            .unwrap();
    }
    let mut f = |n, s| add_fn(c, n, s, ["fp32:0", "fp32:1"], ["fp32:0"]);
    f("fp32.add", FP_32_ADD)?;
    f("fp32.sub", FP_32_SUB)?;
    f("fp32.mul", FP_32_MUL)?;
    f("fp32.div", FP_32_DIV)?;
    let mut f = |n, s| add_fn(c, n, s, ["fp32:0"], ["fp32:0"]);
    f("fp32.sign", FP_32_SIGN)?;
    f("fp32.sqrt", FP_32_SQRT)?;
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

        c.registers.try_insert(arg, ty.clone()).unwrap();
        c.types
            .try_insert(ty, Type::Builtin(BuiltinType::Int(i)))
            .unwrap();
    }

    for a in 0..2 {
        c.registers
            .try_insert(format!("int32:{a}").into(), "Int32".into())
            .unwrap();
    }
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
    add_fn(c, "int32.sign", INT_32_SIGN, ["int32:0"], ["int2:0"])?;
    add_fn(
        c,
        "int32.divmod",
        NAT_32_DIVMOD,
        ["int32:0", "int32:1"],
        ["int32:0", "int32:1"],
    )?;

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

    pub fn handle(
        vm: &WordVM,
        state: &mut WordVMState,
        id: u32,
    ) -> Result<Option<External>, Error> {
        let (input, output) = vm.sys_registers(id);
        let f = |i| vm.constant(i);
        let input = |i| state.register(f(input + i)?);
        let output = |state: &mut WordVMState, i, v| state.set_register(f(output + i)?, v);
        match id {
            CONST_STR_GET => {
                let offt = input(0)?;
                let len = input(1)?;
                let index = input(2)?;
                let c = vm.strings_buffer()[offt as usize..][..len as usize]
                    .get(index as usize)
                    .copied()
                    .unwrap_or(u8::MAX);
                output(state, 0, u32::from(c))?;
            }
            CONST_STR_LEN => {
                let len = input(1)?;
                output(state, 0, len)?;
            }
            INT_32_SUB => output(state, 0, input(0)?.wrapping_sub(input(1)?))?,
            INT_32_SIGN => output(state, 0, (input(0)? as i32).signum() as u32)?,
            NAT_32_DIVMOD => {
                let (x, y) = (input(0)?, input(1)?);
                output(state, 0, x / y)?;
                output(state, 1, x % y)?;
            }
            WRITE_BYTE => return Ok(Some(External::WriteByte(input(0)? as u8))),
            id => match id & !0x1f {
                INT_32_TO_X => output(state, 0, input(0)?)?,
                NAT_X_TO_32 => {
                    let shift = 32 - (id & 0x1f);
                    output(state, 0, (input(0)? << shift) >> shift)?;
                }
                INT_X_TO_32 => {
                    let shift = 32 - (id & 0x1f);
                    output(state, 0, ((input(0)? << shift) as i32 >> shift) as u32)?;
                }
                _ => Err(Error)?,
            },
        }
        Ok(None)
    }
}
