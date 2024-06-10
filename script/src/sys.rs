use crate::{BuiltinType, Collection, Error, ErrorKind, Function, Mode, Type};

const INT_32_TO_X: u32 = 0 << 5;
const INT_X_TO_32: u32 = 1 << 5;

const INT_32_SUB: u32 = 2 << 5 | 0;
const INT_32_SIGN: u32 = 2 << 5 | 1;
const INT_32_DIVMOD: u32 = 2 << 5 | 2;

const CONST_STR_GET: u32 = 2 << 5 | 8;
const CONST_STR_LEN: u32 = 2 << 5 | 9;

const WRITE_BYTE: u32 = 2 << 5 | 15;

pub fn add_defaults(collection: &mut Collection) -> Result<(), Error> {
    add_integer(collection)?;
    add_const_str(collection)?;
    add_write(collection)?;
    Ok(())
}

fn add_integer(c: &mut Collection) -> Result<(), Error> {
    use Mode::*;

    for i in 1..=31 {
        let ty = format!("Int{i}").into_boxed_str();
        let arg = format!("int{i}:0").into_boxed_str();

        add_fn(
            c,
            format!("int{i}.to_32"),
            INT_32_TO_X | u32::from(i),
            [(In, &*arg), (Out, "int32:0")],
        )?;
        add_fn(
            c,
            format!("int32.to_{i}"),
            INT_X_TO_32 | u32::from(i),
            [(In, "int32:0"), (Out, &*arg)],
        )?;

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
        [(InOut, "int32:0"), (InOut, "int32:1")],
    )?;
    add_fn(c, "int32.sign", INT_32_SIGN, [(In, "int32:0"), (Out, "int2:0")])?;
    add_fn(
        c,
        "int32.divmod",
        INT_32_DIVMOD,
        [(InOut, "int32:0"), (InOut, "int32:1")],
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
        [
            (Mode::In, "const_str:0"),
            (Mode::In, "int32:0"),
            (Mode::Out, "int8:0"),
        ],
    )?;
    add_fn(
        c,
        "const_str.len",
        CONST_STR_LEN,
        [(Mode::In, "const_str:0"), (Mode::Out, "int32:0")],
    )?;
    Ok(())
}

fn add_write(c: &mut Collection) -> Result<(), Error> {
    add_fn(c, "write.byte", WRITE_BYTE, [(Mode::In, "int8:0")])
}

fn add_fn(
    collection: &mut Collection,
    name: impl Into<Box<str>>,
    id: u32,
    registers: impl IntoIterator<Item = (Mode, impl Into<Box<str>>)>,
) -> Result<(), Error> {
    _add_fn(
        collection,
        name.into(),
        id,
        registers.into_iter().map(|(m, n)| (m, n.into())).collect(),
    )
}

fn _add_fn(
    collection: &mut Collection,
    name: Box<str>,
    id: u32,
    registers: Box<[(Mode, Box<str>)]>,
) -> Result<(), Error> {
    collection
        .functions
        .try_insert(
            name,
            Function::Builtin {
                id,
                registers: registers
                    .iter()
                    .map(|(m, n)| (*m, n.to_string().into_boxed_str()))
                    .collect(),
            },
        )
        .map(|_| ())
        .map_err(|e| Error {
            line: 0,
            kind: ErrorKind::DuplicateFunction(e.entry.key().clone().into()),
        })
}

pub mod wordvm {
    use {super::*, crate::executor::wordvm::{Error, WordVM, WordVMState}};

    pub enum External {
        WriteByte(u8),
    }

    pub fn handle(vm: &WordVM, state: &mut WordVMState, id: u32) -> Result<Option<External>, Error> {
        let regs = vm.sys_registers(id);
        let f = |i| regs.get(i).copied().ok_or(Error);
        let rr = |i, o| state.register(f(i)? + o);
        let r = |i| rr(i, 0);
        match id {
            CONST_STR_GET => {
                let offt = rr(0, 0)?;
                let len = rr(0, 1)?;
                let index = r(1)?;
                let c = vm.strings_buffer()[offt as usize..][..len as usize]
                    .get(index as usize)
                    .copied()
                    .unwrap_or(u8::MAX);
                state.set_register(*regs.get(2).ok_or(Error)?, u32::from(c))?;
            }
            CONST_STR_LEN => {
                let len = rr(0, 1)?;
                state.set_register(f(1)?, len)?;
            }
            INT_32_SUB => {
                state.set_register(f(0)?, r(0)?.wrapping_sub(r(1)?))?
            }
            INT_32_SIGN => {
                state.set_register(f(1)?, (r(0)? as i32).signum() as u32)?
            }
            WRITE_BYTE => {
                return Ok(Some(External::WriteByte(r(0)? as u8)))
            }
            id => todo!("{id}"),
            _ => Err(Error)?,
        }
        Ok(None)
    }
}
