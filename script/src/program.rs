use {
    crate::{BuiltinType, Collection, Error, ErrorKind, LinearMap, Map, Str},
    core::{fmt, hash::Hash, ops::Index},
    std::rc::Rc,
    util::soa,
};

pub mod debug {
    use crate::Str;

    #[derive(Debug)]
    pub struct Program {
        pub(crate) constants: Box<[Constant]>,
        pub(crate) registers: Box<[Register]>,
        pub(crate) array_registers: Box<[ArrayRegister]>,
        pub(crate) functions: Box<[Function]>,
    }

    #[derive(Debug, Default)]
    pub(crate) struct Constant {
        pub name: Str,
    }

    #[derive(Debug, Default)]
    pub(crate) struct Register {
        pub name: Str,
        pub value: Str,
        pub line: Source,
    }

    #[derive(Debug, Default)]
    pub(crate) struct ArrayRegister {
        pub name: Str,
        pub index: Str,
        pub value: Str,
        pub line: Source,
    }

    #[derive(Debug, Default)]
    pub(crate) struct Function {
        pub name: Str,
        pub instruction_to_line: Box<[Source]>,
        pub last_line: Source,
    }

    #[derive(Clone, Copy, Default, Debug)]
    pub(crate) struct Source {
        pub file: u32,
        pub line: u32,
    }
}

#[derive(Debug)]
pub struct Program {
    pub(crate) constants: Box<[Constant]>,
    pub(crate) registers: Box<[Register]>,
    pub(crate) array_registers: Box<[ArrayRegister]>,
    pub(crate) functions: Box<[Function]>,
    pub(crate) sys_to_registers: Box<[Option<SysRegisterMap>]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Constant {
    Int(u32),
    Str(Box<[u8]>),
    // as bits so we can match exactly with Eq
    Fp(u32),
}

#[derive(Debug)]
pub(crate) enum Register {
    Int(u8),
    Fp32,
    Str,
    User(u32),
}

#[derive(Debug)]
pub(crate) struct ArrayRegister {
    pub value: Register,
    pub dimensions: Rc<[Register]>,
}

#[derive(Clone, Debug)]
enum RegisterMap<'a> {
    Unit { index: u32 },
    Group { fields: LinearMap<&'a Str, Self> },
}

#[derive(Debug)]
pub(crate) enum Function {
    Block(FunctionBlock),
    Switch(FunctionSwitch),
}

#[derive(Debug, Default)]
pub(crate) struct FunctionBlock {
    pub instructions: Box<[Instruction]>,
    pub next: i32,
}

#[derive(Debug)]
pub(crate) struct FunctionSwitch {
    pub register: u32,
    pub cases: Box<[SwitchCase]>,
    pub default: Option<i32>,
}

#[derive(Debug)]
pub(crate) struct SwitchCase {
    pub constant: u32,
    pub function: i32,
}

#[derive(Debug)]
pub(crate) struct SysRegisterMap {
    pub inputs: Box<[u32]>,
    pub outputs: Box<[u32]>,
}

#[derive(Debug, Default)]
struct ProgramBuilder<'a> {
    types: IndexMapBuilder<&'a Str, TypeId, Type<'a>, ()>,
    group_constants: Vec<Option<&'a Map<Str, Map<Str, Str>>>>,
    functions: IndexMapBuilder<&'a Str, i32, Function, debug::Function>,
    constants: IndexMapBuilder<Constant, u32, Constant, debug::Constant>,
    registers: IndexMapBuilder<&'a Str, (RegisterMap<'a>, TypeId), Register, debug::Register>,
    array_registers: IndexMapBuilder<
        &'a Str,
        (RegisterMap<'a>, TypeId, TypeId),
        ArrayRegister,
        debug::ArrayRegister,
    >,
    sys_to_registers: Map<u32, SysRegisterMap>,
}

struct IndexMapBuilder<K, I, T, U> {
    to_index: Map<K, I>,
    values: soa::Vec2<T, U>,
}

#[derive(Debug)]
struct FunctionBuilder<'a, 'b> {
    program: &'b mut ProgramBuilder<'a>,
    instructions: Vec<Instruction>,
}

#[derive(Debug)]
enum Type<'a> {
    Int(u8),
    Fp32,
    ConstantString,
    Opaque { bits: u8 },
    Enum { value_to_id: Map<&'a Str, u32> },
    Group { fields: LinearMap<&'a Str, TypeId> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TypeId(u32);

#[derive(Clone, Copy)]
pub(crate) enum Instruction {
    /// Load a register value.
    RegisterLoad(u32),
    /// Store a register value.
    RegisterStore(u32),

    /// Load a function.
    ConstantLoad(u32),

    /// Start an array access.
    ArrayAccess(u32),
    /// Perform an index operation with the given register.
    ArrayIndex(u32),
    /// Store a register value into the array.
    ArrayStore(u32),

    /// Call a function
    ///
    /// If negative, it refers to a builtin function.
    Call(i32),
}

enum Int32 {
    U(u32),
    S(i32),
}

impl Program {
    pub fn from_collection(
        collection: &Collection,
        entry: &Str,
    ) -> Result<(Self, debug::Program), Error> {
        let mut builder = ProgramBuilder::default();

        let stub_entry = "".to_string().into_boxed_str();

        builder.collect_types(collection)?;
        builder.collect_constants(collection)?;
        builder.collect_registers(collection)?;
        builder.collect_array_registers(collection)?;
        builder.collect_functions(collection, entry, &stub_entry)?;

        let len = builder
            .sys_to_registers
            .keys()
            .map(|&k| usize::try_from(k).unwrap())
            .max()
            .map_or(0, |x| x + 1);
        let mut sys_to_registers = (0..len).map(|_| None).collect::<Box<[_]>>();

        for (k, v) in builder.sys_to_registers {
            sys_to_registers[usize::try_from(k).unwrap()] = Some(v);
        }

        fn f<T, U>(v: soa::Vec2<T, U>) -> (Box<[T]>, Box<[U]>) {
            let (t, u) = v.into();
            (t.into_boxed_slice(), u.into_boxed_slice())
        }
        let (constants, debug_constants) = f(builder.constants.values);
        let (registers, debug_registers) = f(builder.registers.values);
        let (array_registers, debug_array_registers) = f(builder.array_registers.values);
        let (functions, debug_functions) = f(builder.functions.values);

        for (f, d) in functions.iter().zip(debug_functions.iter()) {
            match f {
                Function::Block(b) => assert_eq!(b.instructions.len(), d.instruction_to_line.len()),
                Function::Switch(s) => assert_eq!(s.cases.len(), d.instruction_to_line.len()),
            }
        }

        let program = Self {
            constants,
            registers,
            array_registers,
            functions,
            sys_to_registers,
        };

        let debug = debug::Program {
            constants: debug_constants,
            registers: debug_registers,
            array_registers: debug_array_registers,
            functions: debug_functions,
        };

        Ok((program, debug))
    }

    pub fn max_call_depth(&self) -> u32 {
        let mut visited = util::bit::BitVec::filled(self.functions.len(), false);
        self._max_call_depth(0, &mut visited)
    }

    fn _max_call_depth(&self, entry: i32, visited: &mut util::bit::BitVec) -> u32 {
        if visited.get(entry as usize).unwrap() {
            // FIXME
            return 0;
        }
        visited.set(entry as usize, true);
        let n = match &self.functions[entry as usize] {
            Function::Block(f) => f
                .instructions
                .iter()
                .flat_map(|i| match i {
                    &Instruction::Call(address) if address >= 0 => Some((address, 1)),
                    _ => None,
                })
                .chain((f.next >= 0).then_some((f.next, 0)))
                .map(|(a, n)| n + self._max_call_depth(a, visited))
                .max()
                .unwrap_or(0),
            Function::Switch(s) => s
                .cases
                .iter()
                .map(|c| c.function)
                .chain(s.default)
                .filter(|&a| a >= 0)
                .map(|a| self._max_call_depth(a, visited))
                .max()
                .unwrap_or(0),
        };
        visited.set(entry as usize, false);
        n
    }
}

impl Program {
    pub fn to_bytes(&self, abi: u128) -> Vec<u8> {
        todo!()
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegisterLoad(from) => write!(f, "R.LOAD  {from}"),
            Self::RegisterStore(to) => write!(f, "R.STORE {to}"),
            Self::ConstantLoad(from) => write!(f, "C.LOAD  {from}"),
            Self::ArrayAccess(array) => write!(f, "A.ACCES {array}"),
            Self::ArrayIndex(index) => write!(f, "A.INDEX {index}"),
            Self::ArrayStore(register) => write!(f, "A.STORE {register}"),
            Self::Call(address) => write!(f, "CALL    {address}"),
        }
    }
}

impl<'a> ProgramBuilder<'a> {
    fn collect_types(&mut self, collection: &'a Collection) -> Result<(), Error> {
        for (i, (name, _)) in collection.types.iter().enumerate() {
            let i = u32::try_from(i).unwrap();
            self.types.to_index.try_insert(name, TypeId(i)).unwrap();
        }

        for (_, ty) in collection.types.iter() {
            let ty = match ty {
                crate::Type::Enum(values) => {
                    let value_to_id = values
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (v, u32::try_from(i).unwrap()))
                        .collect();
                    Type::Enum { value_to_id }
                }
                crate::Type::Group(fields) => {
                    let fields = fields
                        .iter()
                        .map(|(name, ty)| (name, self.types.to_index[ty]))
                        .collect();
                    Type::Group { fields }
                }
                &crate::Type::Builtin(b) => match b {
                    BuiltinType::Int(bits) => Type::Int(bits),
                    BuiltinType::Fp32 => Type::Fp32,
                    BuiltinType::ConstantString => Type::ConstantString,
                    BuiltinType::Opaque { bits } => Type::Opaque { bits },
                },
            };
            self.types.values.push((ty, ()));
        }
        Ok(())
    }

    fn collect_constants(&mut self, collection: &'a Collection) -> Result<(), Error> {
        self.group_constants.resize(self.types.values.len(), None);
        for (ty, consts) in collection.constants.iter() {
            let ty = self.types.to_index[ty];
            self.group_constants[ty.0 as usize] = Some(consts);
        }
        Ok(())
    }

    fn collect_registers(&mut self, collection: &'a Collection) -> Result<(), Error> {
        for (name, ty_s) in collection.registers.iter() {
            let ty = *self
                .types
                .to_index
                .get(ty_s)
                .ok_or_else(|| todo!("{ty_s}"))?;
            let regmap = self.expand_register_group(ty, name, ty_s, 1337, 1337)?;
            self.registers
                .to_index
                .try_insert(name, (regmap, ty))
                .unwrap();
        }
        Ok(())
    }

    fn expand_register_group(
        &mut self,
        mut ty: TypeId,
        name: &'a Str,
        value_ty: &'a Str,
        file: u32,
        line: u32,
    ) -> Result<RegisterMap<'a>, Error> {
        // fuck this mutable borrow shit
        // per-field borrowing when?
        let mut stack = Vec::new();

        let regmap = 'l: loop {
            match &self.types[ty] {
                Type::Group { fields } => {
                    let mut it = fields.iter().map(|(a, b)| (*a, *b));
                    if let Some((name, ty_f)) = it.next() {
                        ty = ty_f;
                        stack.push((LinearMap::default(), it, name));
                    }
                    continue 'l;
                }
                tyty => {
                    let index = u32::try_from(self.registers.values.len()).unwrap();
                    let debug = debug::Register {
                        name: name.clone(),
                        value: value_ty.clone(),
                        line: debug::Source { file, line },
                    };
                    self.registers.values.push((tyty.to_register(), debug));

                    let mut regmap = RegisterMap::Unit { index };

                    while let Some((mut fields, mut it, name)) = stack.pop() {
                        fields.try_insert(name, regmap).unwrap();
                        if let Some((name, ty_f)) = it.next() {
                            ty = ty_f;
                            stack.push((fields, it, name));
                            continue 'l;
                        } else {
                            regmap = RegisterMap::Group { fields }
                        }
                    }

                    break 'l regmap;
                }
            };
        };

        Ok(regmap)
    }

    fn collect_array_registers(&mut self, collection: &'a Collection) -> Result<(), Error> {
        for (name, (index_ty, value_ty)) in collection.array_registers.iter() {
            let index = self.types.to_index[index_ty];
            let value = self.types.to_index[value_ty];
            let dimensions = self.type_index_dimensions(index).unwrap().into();
            let regmap = self.expand_array_register_group(
                value, dimensions, name, index_ty, value_ty, 1337, 1337,
            )?;
            self.array_registers
                .to_index
                .try_insert(name, (regmap, index, value))
                .unwrap();
        }
        Ok(())
    }

    fn expand_array_register_group(
        &mut self,
        mut ty: TypeId,
        dimensions: Rc<[Register]>,
        name: &'a Str,
        index_ty: &'a Str,
        // TODO derive
        value_ty: &'a Str,
        file: u32,
        line: u32,
    ) -> Result<RegisterMap<'a>, Error> {
        // fuck this mutable borrow shit
        // per-field borrowing when?
        let mut stack = Vec::new();

        let regmap = 'l: loop {
            match &self.types[ty] {
                Type::Group { fields } => {
                    let mut it = fields.iter().map(|(a, b)| (*a, *b));
                    if let Some((name, ty_f)) = it.next() {
                        ty = ty_f;
                        stack.push((LinearMap::default(), it, name));
                    }
                    continue 'l;
                }
                tyty => {
                    let index = u32::try_from(self.array_registers.values.len()).unwrap();
                    let array = ArrayRegister {
                        value: tyty.to_register(),
                        dimensions: dimensions.clone(),
                    };
                    let debug = {
                        debug::ArrayRegister {
                            name: name.clone(),
                            index: index_ty.clone(),
                            value: value_ty.clone(),
                            line: debug::Source { file, line },
                        }
                    };
                    self.array_registers.values.push((array, debug));

                    let mut regmap = RegisterMap::Unit { index };

                    while let Some((mut fields, mut it, name)) = stack.pop() {
                        fields.try_insert(name, regmap).unwrap();
                        if let Some((name, ty_f)) = it.next() {
                            ty = ty_f;
                            stack.push((fields, it, name));
                            continue 'l;
                        } else {
                            regmap = RegisterMap::Group { fields }
                        }
                    }

                    break 'l regmap;
                }
            };
        };

        Ok(regmap)
    }

    fn collect_functions(
        &mut self,
        collection: &'a Collection,
        entry: &'a Str,
        entry_stub: &'a Str,
    ) -> Result<(), Error> {
        // prepass so we can immediately resolve addresses
        let mut i = 0;
        // insert stub function as entry
        // it's an easy hack and is trivial to optimize anyway
        self.functions.to_index.try_insert(entry_stub, i).unwrap();
        i += 1;
        for (name, f) in collection.functions.iter() {
            match f {
                crate::Function::User { .. } => {
                    self.functions.to_index.try_insert(name, i).unwrap();
                    i += 1;
                }
                crate::Function::Builtin { id, .. } => {
                    let id = !i32::try_from(*id).unwrap();
                    self.functions.to_index.try_insert(name, id).unwrap();
                }
            }
        }
        for (name, _) in collection.switches.iter() {
            self.functions.to_index.try_insert(name, i).unwrap();
            i += 1;
        }

        // entry stub
        {
            let f = FunctionBlock {
                instructions: Default::default(),
                next: self.function(entry)?,
            };
            let debug = debug::Function {
                name: entry_stub.clone(),
                instruction_to_line: [].into(),
                last_line: Default::default(),
            };
            self.functions.values.push((Function::Block(f), debug));
        }

        for (name, func) in collection.functions.iter() {
            match func {
                crate::Function::User {
                    instructions: instrs,
                    file: f,
                    last_line: ll,
                    lines,
                    next,
                } => {
                    let last_line = *ll;
                    let src = |line| debug::Source { file: *f, line };
                    let mut instruction_to_line = Vec::new();
                    let mut builder = FunctionBuilder {
                        program: self,
                        instructions: Default::default(),
                    };

                    for (instr, &line) in instrs.iter().zip(lines.iter()) {
                        match instr {
                            crate::Instruction::Set { to, value } => builder.set(to, value)?,
                            crate::Instruction::Move { to, from } => builder.move_(to, from)?,
                            crate::Instruction::ToArray {
                                index,
                                array,
                                register,
                            } => builder.to_array(index, array, register)?,
                            crate::Instruction::FromArray {
                                index,
                                array,
                                register,
                            } => builder.from_array(index, array, register)?,
                            crate::Instruction::Call { function } => builder.call(function)?,
                            crate::Instruction::ToGroup {
                                group,
                                field,
                                register,
                            } => builder.to_group(group, field, register)?,
                            crate::Instruction::FromGroup {
                                group,
                                field,
                                register,
                            } => builder.from_group(group, field, register)?,
                        }
                        instruction_to_line.resize(builder.instructions.len(), src(line));
                    }

                    let f = builder.finish(next)?;

                    let debug = debug::Function {
                        name: name.clone(),
                        instruction_to_line: instruction_to_line.into(),
                        last_line: src(last_line),
                    };
                    self.functions.values.push((Function::Block(f), debug));
                }
                crate::Function::Builtin {
                    id,
                    inputs,
                    outputs,
                } => {
                    let id = *id;
                    let f = |s: &[Str]| {
                        let mut v = Vec::new();
                        fn rec(v: &mut Vec<u32>, r: &RegisterMap<'_>) -> Result<(), Error> {
                            match r {
                                RegisterMap::Unit { index } => v.push(*index),
                                RegisterMap::Group { fields } => {
                                    for rr in fields.values() {
                                        rec(v, rr)?
                                    }
                                }
                            }
                            Ok(())
                        }
                        for r in s.iter() {
                            let (reg, _) = self.register(r)?;
                            rec(&mut v, reg)?;
                        }
                        Ok(v.into())
                    };
                    let map = SysRegisterMap {
                        inputs: f(inputs)?,
                        outputs: f(outputs)?,
                    };
                    self.sys_to_registers.try_insert(id, map).map_err(|e| {
                        dbg!(e.entry.key());
                        todo!();
                    })?;
                }
            };
        }

        for (name, switch) in collection.switches.iter() {
            let (register, ty) = self.register(&switch.register)?;
            match &self.types[ty] {
                Type::ConstantString => todo!(),
                Type::Int(_) | Type::Enum { .. } => {
                    let &RegisterMap::Unit { index: register } = register else {
                        todo!()
                    };
                    let mut cases = Vec::new();
                    for (value, next) in switch.branches.iter() {
                        let constant = self.get_or_add_const(ty, value)?;
                        let function = self.function(next)?;
                        cases.push(SwitchCase { constant, function });
                    }
                    let default = switch
                        .default
                        .as_ref()
                        .map(|s| self.function(s))
                        .transpose()?;
                    let f = FunctionSwitch {
                        register,
                        cases: cases.into(),
                        default,
                    };
                    let src = |line| debug::Source {
                        file: switch.file,
                        line,
                    };
                    let debug = debug::Function {
                        name: name.clone(),
                        instruction_to_line: switch.lines.iter().map(|&l| src(l)).collect(),
                        last_line: src(switch.last_line),
                    };
                    self.functions.values.push((Function::Switch(f), debug));
                }
                Type::Fp32 => todo!(),
                Type::Opaque { .. } => todo!(),
                Type::Group { .. } => todo!(),
            }
        }

        Ok(())
    }

    fn register(&self, name: &'a Str) -> Result<(&RegisterMap<'a>, TypeId), Error> {
        self.registers
            .to_index
            .get(name)
            .map(|(x, y)| (x, *y))
            .ok_or_else(|| Error {
                kind: ErrorKind::RegisterNotFound(name.to_string()),
                line: 0,
            })
    }

    fn array_register(&self, name: &'a Str) -> Result<(&RegisterMap<'a>, TypeId, TypeId), Error> {
        self.array_registers
            .to_index
            .get(name)
            .map(|(x, y, z)| (x, *y, *z))
            .ok_or_else(|| Error {
                kind: ErrorKind::RegisterNotFound(name.to_string()),
                line: 0,
            })
    }

    fn function(&self, name: &Str) -> Result<i32, Error> {
        self.functions
            .to_index
            .get(name)
            .ok_or_else(|| Error {
                kind: ErrorKind::FunctionNotFound(name.to_string()),
                line: 0,
            })
            .copied()
    }

    fn get_or_add_const(&mut self, ty: TypeId, value: &'a Str) -> Result<u32, Error> {
        let val = {
            match &self.types[ty] {
                Type::Int(bits) | Type::Opaque { bits } => {
                    Constant::Int(self.parse_integer(value, *bits)?)
                }
                // TODO handle underscores
                Type::Fp32 => Constant::Fp(value.parse::<f32>().unwrap().to_bits()),
                Type::ConstantString => Constant::Str(self.parse_constant_string(value)?),
                Type::Enum { value_to_id } => Constant::Int(value_to_id[value]),
                Type::Group { .. } => todo!("group constants?"),
            }
        };

        let i = self
            .constants
            .to_index
            .entry(val.clone())
            .or_insert_with(|| {
                let i = u32::try_from(self.constants.values.len()).unwrap();
                let debug = debug::Constant {
                    name: value.clone(),
                };
                self.constants.values.push((val, debug));
                i
            });

        Ok(*i)
    }

    fn parse_constant_string(&mut self, value: &Str) -> Result<Box<[u8]>, Error> {
        let mut it = value.bytes();
        if it.next() != Some(b'"') {
            todo!();
        }

        let mut s = Vec::new();
        loop {
            let c = it.next().unwrap();
            match c {
                b'\\' => todo!(),
                b'"' => break,
                c => s.push(c),
            }
        }

        if it.next().is_some() {
            todo!();
        }

        Ok(s.into())
    }

    fn parse_integer(&mut self, value: &Str, bits: u8) -> Result<u32, Error> {
        let shift = 32 - u32::from(bits);
        let max = u32::MAX >> shift;
        let min = i32::MIN >> shift;
        match self.parse_integer32(value)? {
            Int32::U(n) => {
                if n > max {
                    todo!("{n} > {max}");
                }
                Ok(n)
            }
            Int32::S(n) => {
                if n < min {
                    todo!("{n} < {min}");
                }
                Ok(n as u32 & max)
            }
        }
    }

    fn parse_integer32(&mut self, value: &Str) -> Result<Int32, Error> {
        let (neg, s) = if value.starts_with("-") {
            (true, &value[1..])
        } else {
            (false, &value[..])
        };
        let n = self.parse_natural128(s)?;
        let n = i128::try_from(n).unwrap();
        match neg {
            true => Ok(Int32::S((n as i32).wrapping_neg())),
            false => Ok(Int32::U(n as u32)),
        }
    }

    fn parse_char(&mut self, value: &str) -> Result<u32, Error> {
        let mut it = value.chars();
        if it.next() != Some('\'') {
            todo!();
        }

        let c = match it.next() {
            Some('\\') => match it.next() {
                Some('n') => b'\n' as u32,
                c => todo!("{:?}", c),
            },
            Some('\'') => todo!(),
            Some(c) => c as u32,
            None => todo!(),
        };

        if it.next() != Some('\'') {
            todo!();
        }
        if it.next().is_some() {
            todo!();
        }

        Ok(c)
    }

    fn parse_natural128(&mut self, value: &str) -> Result<u128, Error> {
        if value.starts_with("'") {
            return self.parse_char(value).map(u128::from);
        }

        let base = 10;

        let mut n = 0u128;
        for b in value.bytes() {
            let x = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                c => todo!("{}", c as char),
            };
            let x = u128::from(x);
            if x >= base {
                dbg!(x as u8 as char);
                todo!();
            }
            n = n.checked_mul(base).ok_or_else(|| todo!())?;
            n = n.checked_add(x).ok_or_else(|| todo!())?;
        }

        Ok(n)
    }

    fn type_index_dimensions(&self, ty: TypeId) -> Option<Vec<Register>> {
        let ty = &self.types[ty];
        match ty {
            // Don't allow indexing on opaque since
            // - the bit length may change at any time
            // - opaque values may represent handles,
            //   and the value referenced may change without the opaque handle itself changing
            Type::Fp32 | Type::ConstantString | Type::Opaque { .. } => None,
            Type::Int(bits) if *bits >= 32 => None,
            Type::Int(_) | Type::Enum { .. } => Some([ty.to_register()].into()),
            Type::Group { fields } => {
                let mut dim = Vec::new();
                for (_, &v) in fields.iter() {
                    dim.extend(self.type_index_dimensions(v)?);
                }
                Some(dim)
            }
        }
    }
}

impl<'a, 'b> FunctionBuilder<'a, 'b> {
    fn set(&mut self, to: &'a Str, value: &'a Str) -> Result<(), Error> {
        let (reg, ty) = self.program.register(to)?;
        // FIXME
        let reg = reg.clone();
        self.set_recursive(&reg, ty, value)
    }

    fn set_recursive(
        &mut self,
        reg: &RegisterMap<'a>,
        ty: TypeId,
        value: &'a Str,
    ) -> Result<(), Error> {
        match reg {
            &RegisterMap::Unit { index: to } => {
                let from = self.program.get_or_add_const(ty, value)?;
                self.instructions.push(Instruction::ConstantLoad(from));
                self.instructions.push(Instruction::RegisterStore(to));
            }
            RegisterMap::Group { fields } => {
                let csts = &self.program.group_constants[ty.0 as usize].unwrap();
                let csts = &csts[value];
                assert_eq!(fields.len(), csts.len());

                let Type::Group { fields: ty_fields } = &self.program.types[ty] else {
                    todo!()
                };

                //FIXME
                let ty_fields = ty_fields.clone();

                for (k, v) in csts.iter() {
                    let f_reg = &fields[k];
                    let f_ty = ty_fields[k];
                    self.set_recursive(f_reg, f_ty, v)?;
                }
            }
        }
        Ok(())
    }

    fn move_(&mut self, to: &'a Str, from: &'a Str) -> Result<(), Error> {
        let (to, to_ty) = self.program.register(to)?;
        let (from, from_ty) = self.program.register(from)?;
        if to_ty != from_ty {
            dbg!(&self.program.types, to_ty, from_ty);
            todo!()
        }

        Self::move_recursive(&mut self.instructions, to, from)
    }

    fn move_recursive(
        instructions: &mut Vec<Instruction>,
        to: &RegisterMap<'_>,
        from: &RegisterMap<'_>,
    ) -> Result<(), Error> {
        use RegisterMap::*;
        match (to, from) {
            (&Unit { index: to }, &Unit { index: from }) => {
                instructions.push(Instruction::RegisterLoad(from));
                instructions.push(Instruction::RegisterStore(to));
            }
            (Group { fields: to }, Group { fields: from }) => {
                if to.len() != from.len() {
                    dbg!(to, from);
                    todo!();
                }
                for (k, f_to) in to.iter() {
                    let f_from = &from[k];
                    Self::move_recursive(instructions, f_to, f_from)?;
                }
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn to_array(&mut self, index: &'a Str, array: &'a Str, register: &'a Str) -> Result<(), Error> {
        let (index, index_ty) = self.program.register(index)?;
        let (array, array_index_ty, array_ty) = self.program.array_register(array)?;
        let (register, register_ty) = self.program.register(register)?;

        assert_eq!(index_ty, array_index_ty);
        assert_eq!(array_ty, register_ty);

        Self::array_recursive(&mut self.instructions, true, index, array, register)
    }

    fn from_array(
        &mut self,
        index: &'a Str,
        array: &'a Str,
        register: &'a Str,
    ) -> Result<(), Error> {
        let (index, index_ty) = self.program.register(index)?;
        let (array, array_index_ty, array_ty) = self.program.array_register(array)?;
        let (register, register_ty) = self.program.register(register)?;

        assert_eq!(index_ty, array_index_ty);
        assert_eq!(array_ty, register_ty);

        Self::array_recursive(&mut self.instructions, false, index, array, register)
    }

    fn array_recursive(
        instructions: &mut Vec<Instruction>,
        is_to: bool,
        index: &RegisterMap<'a>,
        array: &RegisterMap<'a>,
        register: &RegisterMap<'a>,
    ) -> Result<(), Error> {
        use RegisterMap::*;
        match (array, register) {
            (&Unit { index: array }, &Unit { index: register }) => {
                instructions.push(Instruction::ArrayAccess(array));
                Self::array_recursive_index(instructions, index)?;
                instructions.push(if is_to {
                    Instruction::ArrayStore(register)
                } else {
                    Instruction::RegisterStore(register)
                });
            }
            (Group { fields: array }, Group { fields: register }) => {
                for ((k1, array), (k2, register)) in array.iter().zip(register.iter()) {
                    debug_assert_eq!(k1, k2);
                    Self::array_recursive(instructions, is_to, index, array, register)?;
                }
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn array_recursive_index(
        instructions: &mut Vec<Instruction>,
        index: &RegisterMap<'a>,
    ) -> Result<(), Error> {
        match index {
            &RegisterMap::Unit { index } => instructions.push(Instruction::ArrayIndex(index)),
            RegisterMap::Group { fields } => {
                for (_, v) in fields.iter() {
                    Self::array_recursive_index(instructions, v)?
                }
            }
        }
        Ok(())
    }

    fn call(&mut self, function: &'a Str) -> Result<(), Error> {
        let address = self.program.function(function)?;
        self.instructions.push(Instruction::Call(address));
        Ok(())
    }

    fn to_group(&mut self, group: &'a Str, field: &'a Str, register: &'a Str) -> Result<(), Error> {
        self.group(true, group, field, register)
    }

    fn from_group(
        &mut self,
        group: &'a Str,
        field: &'a Str,
        register: &'a Str,
    ) -> Result<(), Error> {
        self.group(false, group, field, register)
    }

    fn group(
        &mut self,
        is_to: bool,
        group: &'a Str,
        field: &'a Str,
        register: &'a Str,
    ) -> Result<(), Error> {
        let (group_reg, _) = self.program.register(group)?;
        let (reg, _) = self.program.register(register)?;

        let RegisterMap::Group { fields } = group_reg else {
            todo!()
        };
        let field_reg = &fields[field];

        // FIXME check type
        let (to, from) = if is_to {
            (field_reg, reg)
        } else {
            (reg, field_reg)
        };

        Self::move_recursive(&mut self.instructions, to, from)
    }

    fn finish(self, next: &'a Option<Str>) -> Result<FunctionBlock, Error> {
        let next = next
            .as_ref()
            .map(|n| self.program.function(n))
            .transpose()?
            .unwrap_or(-1);
        Ok(FunctionBlock {
            instructions: self.instructions.into(),
            next,
        })
    }
}

impl<'a> Type<'a> {
    /// Minimum size in bits.
    pub fn bit_size(&self) -> u32 {
        match self {
            Self::Int(bits) => (*bits).into(),
            Self::Fp32 => 32,
            // offset + length
            Self::ConstantString => 32 + 32,
            Self::Opaque { bits } => (*bits).into(),
            Self::Enum { value_to_id } => value_to_id
                .len()
                .checked_next_power_of_two()
                .map_or(usize::BITS, |x| x.try_into().unwrap()),
            Self::Group { fields: _ } => todo!(),
        }
    }
}

impl Default for Function {
    fn default() -> Self {
        Self::Block(Default::default())
    }
}

impl SysRegisterMap {
    pub fn iter_all(&self) -> impl Iterator<Item = u32> + '_ {
        [&self.inputs, &self.outputs]
            .into_iter()
            .flat_map(|s| s.iter().copied())
    }
    pub fn iter_all_mut(&mut self) -> impl Iterator<Item = &mut u32> + '_ {
        [&mut self.inputs, &mut self.outputs]
            .into_iter()
            .flat_map(|s| s.iter_mut())
    }
}

impl<K, I, T, U> Default for IndexMapBuilder<K, I, T, U> {
    fn default() -> Self {
        Self {
            to_index: Default::default(),
            values: Default::default(),
        }
    }
}

impl<'a, K, T, U> IndexMapBuilder<K, u32, T, U> {
    fn push(&mut self, value: T, debug: U) {
        self.values.push((value, debug));
    }
}

impl<'a, K, T, U> IndexMapBuilder<K, u32, T, U>
where
    K: Eq + Hash,
{
    fn insert(&mut self, key: K, value: T, debug: U) {
        let i = u32::try_from(self.values.len()).unwrap();
        self.to_index.try_insert(key, i).unwrap_or_else(|_| todo!());
        self.push(value, debug);
    }
}

impl<'a, T, U> Index<u32> for IndexMapBuilder<&'a Str, u32, T, U> {
    type Output = T;

    fn index(&self, index: u32) -> &Self::Output {
        self.values.get(index as usize).unwrap().0
    }
}

impl<'a, T, U> Index<TypeId> for IndexMapBuilder<&'a Str, TypeId, T, U> {
    type Output = T;

    fn index(&self, index: TypeId) -> &Self::Output {
        self.values.get(index.0 as usize).unwrap().0
    }
}

impl<K, I, T, U> fmt::Debug for IndexMapBuilder<K, I, T, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TODO")
    }
}

impl Register {
    pub fn bits(&self) -> u8 {
        match self {
            Self::Int(b) => *b,
            Self::Fp32 => 32,
            Self::Str => 64,
            Self::User(n) => u32range_to_bits(*n),
        }
    }

    pub fn range(&self) -> Option<u32> {
        match self {
            &Self::Int(b) => 1u32.checked_shl(b.into()),
            Self::Fp32 => None,
            Self::Str => None,
            Self::User(n) => Some(*n),
        }
    }
}

impl<'a> Type<'a> {
    fn to_register(&self) -> Register {
        match self {
            Self::Int(b) => Register::Int(*b),
            Self::Fp32 => Register::Fp32,
            Self::ConstantString => Register::Str,
            Self::Opaque { bits } => Register::Int(*bits),
            Self::Enum { value_to_id } => Register::User(value_to_id.len() as u32),
            Self::Group { .. } => todo!(),
        }
    }
}

fn u32range_to_bits(max: u32) -> u8 {
    let mut n = max;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n.trailing_ones() as u8
}
