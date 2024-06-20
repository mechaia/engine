use {
    crate::{BuiltinType, Collection, Error, ErrorKind, LinearMap, Map, Str},
    core::fmt,
    std::rc::Rc,
};

#[derive(Debug)]
pub struct Program {
    pub(crate) constants: Box<[Constant]>,
    pub(crate) registers: Box<[Register]>,
    pub(crate) array_registers: Box<[ArrayRegister]>,
    pub(crate) functions: Box<[Function]>,
    pub(crate) sys_to_registers: Box<[Option<SysRegisterMap>]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Constant {
    Int(u32),
    Str(Box<[u8]>),
    // as bits so we can match exactly with Eq
    Fp(u32),
}

#[derive(Debug)]
pub struct Register {
    pub bits: u32,
}

#[derive(Debug)]
pub struct ArrayRegister {
    pub bits: u32,
    pub dimensions: Rc<[u32]>,
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
    pub next: Option<u32>,
}

#[derive(Debug)]
pub(crate) struct FunctionSwitch {
    pub register: u32,
    pub cases: Box<[SwitchCase]>,
    pub default: Option<u32>,
}

#[derive(Debug)]
pub(crate) struct SwitchCase {
    pub constant: u32,
    pub function: u32,
}

#[derive(Debug)]
pub(crate) struct SysRegisterMap {
    pub inputs: Box<[u32]>,
    pub outputs: Box<[u32]>,
}

#[derive(Debug, Default)]
struct ProgramBuilder<'a> {
    types_to_index: Map<&'a Str, TypeId>,
    types: Vec<Type<'a>>,
    group_constants: Vec<Option<&'a Map<Str, Map<Str, Str>>>>,
    function_to_index: Map<&'a Str, u32>,
    functions: Vec<Function>,
    constant_to_index: Map<Constant, u32>,
    constants: Vec<Constant>,
    register_to_index: Map<&'a Str, (RegisterMap<'a>, TypeId)>,
    registers: Vec<Register>,
    array_register_to_index: Map<&'a Str, (RegisterMap<'a>, TypeId, TypeId)>,
    array_registers: Vec<ArrayRegister>,
    sys_to_registers: Map<u32, SysRegisterMap>,
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
    /// Copy one register to another.
    Move { to: u32, from: u32 },
    /// Set register value based on value in constants map.
    Set { to: u32, from: u32 },
    /// Begin array access.
    ArrayAccess { array: u32 },
    /// Index next for current array access.
    ArrayIndex { index: u32 },
    /// Copy a register value to an array element.
    ///
    /// Ends the current array access.
    ArrayStore { register: u32 },
    /// Copy an array element value to a register.
    ///
    /// Ends the current array access.
    ArrayLoad { register: u32 },
    /// Jump and push return address.
    Call { address: u32 },
    /// Call a builtin function.
    Sys { id: u32 },
}

enum Int32 {
    U(u32),
    S(i32),
}

impl Program {
    pub fn from_collection(collection: &Collection, entry: &Str) -> Result<Self, Error> {
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

        Ok(Self {
            constants: builder.constants.into(),
            registers: builder.registers.into(),
            array_registers: builder.array_registers.into(),
            functions: builder.functions.into(),
            sys_to_registers,
        })
    }

    pub fn max_call_depth(&self) -> u32 {
        let mut visited = util::bit::BitVec::filled(self.functions.len(), false);
        self._max_call_depth(0, &mut visited)
    }

    fn _max_call_depth(&self, entry: u32, visited: &mut util::bit::BitVec) -> u32 {
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
                    &Instruction::Call { address } => Some((address, 1)),
                    _ => None,
                })
                .chain(f.next.map(|a| (a, 0)))
                .map(|(a, n)| n + self._max_call_depth(a, visited))
                .max()
                .unwrap_or(0),
            Function::Switch(s) => {
                s.cases
                    .iter()
                    .map(|c| self._max_call_depth(c.function, visited))
                    .max()
                    .unwrap_or(0)
                    + 1
            }
        };
        visited.set(entry as usize, false);
        n
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Move { to, from } => write!(f, "MOVE    {to} {from}"),
            Self::Set { to, from } => write!(f, "SET     {to} {from}"),
            Self::ArrayAccess { array } => write!(f, "A.ACCES {array}"),
            Self::ArrayIndex { index } => write!(f, "A.INDEX {index}"),
            Self::ArrayStore { register } => write!(f, "A.STORE {register}"),
            Self::ArrayLoad { register } => write!(f, "A.LOAD  {register}"),
            Self::Call { address } => write!(f, "CALL    {address}"),
            Self::Sys { id } => write!(f, "SYS     {id:?}"),
        }
    }
}

impl<'a> ProgramBuilder<'a> {
    fn collect_types(&mut self, collection: &'a Collection) -> Result<(), Error> {
        for (i, (name, _)) in collection.types.iter().enumerate() {
            let i = u32::try_from(i).unwrap();
            self.types_to_index.try_insert(name, TypeId(i)).unwrap();
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
                        .map(|(name, ty)| (name, self.types_to_index[ty]))
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
            self.types.push(ty);
        }
        Ok(())
    }

    fn collect_constants(&mut self, collection: &'a Collection) -> Result<(), Error> {
        self.group_constants.resize(self.types.len(), None);
        for (ty, consts) in collection.constants.iter() {
            let ty = self.types_to_index[ty];
            self.group_constants[ty.0 as usize] = Some(consts);
        }
        Ok(())
    }

    fn collect_registers(&mut self, collection: &'a Collection) -> Result<(), Error> {
        for (name, ty) in collection.registers.iter() {
            let ty = *self.types_to_index.get(ty).ok_or_else(|| todo!("{ty}"))?;
            let regmap = self.expand_register_group(ty)?;
            self.register_to_index
                .try_insert(name, (regmap, ty))
                .unwrap();
        }
        Ok(())
    }

    fn expand_register_group(&mut self, mut ty: TypeId) -> Result<RegisterMap<'a>, Error> {
        // fuck this mutable borrow shit
        // per-field borrowing when?
        let mut stack = Vec::new();

        let regmap = 'l: loop {
            match &self.types[usize::try_from(ty.0).unwrap()] {
                Type::Group { fields } => {
                    let mut it = fields.iter().map(|(a, b)| (*a, *b));
                    if let Some((name, ty_f)) = it.next() {
                        ty = ty_f;
                        stack.push((LinearMap::default(), it, name));
                    }
                    continue 'l;
                }
                tyty => {
                    let index = u32::try_from(self.registers.len()).unwrap();
                    let bits = tyty.bit_size();
                    self.registers.push(Register { bits });

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
            let index_ty = self.types_to_index[index_ty];
            let value_ty = self.types_to_index[value_ty];
            let dimensions = self.type_index_dimensions(index_ty).unwrap().into();
            let regmap = self.expand_array_register_group(value_ty, dimensions)?;
            self.array_register_to_index
                .try_insert(name, (regmap, index_ty, value_ty))
                .unwrap();
        }
        Ok(())
    }

    fn expand_array_register_group(
        &mut self,
        mut ty: TypeId,
        dimensions: Rc<[u32]>,
    ) -> Result<RegisterMap<'a>, Error> {
        // fuck this mutable borrow shit
        // per-field borrowing when?
        let mut stack = Vec::new();

        let regmap = 'l: loop {
            match &self.types[usize::try_from(ty.0).unwrap()] {
                Type::Group { fields } => {
                    let mut it = fields.iter().map(|(a, b)| (*a, *b));
                    if let Some((name, ty_f)) = it.next() {
                        ty = ty_f;
                        stack.push((LinearMap::default(), it, name));
                    }
                    continue 'l;
                }
                tyty => {
                    let index = u32::try_from(self.array_registers.len()).unwrap();
                    let bits = tyty.bit_size();
                    let dimensions = dimensions.clone();
                    self.array_registers
                        .push(ArrayRegister { bits, dimensions });

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
        self.function_to_index.try_insert(entry_stub, i).unwrap();
        i += 1;
        for (name, _) in collection.functions.iter() {
            self.function_to_index.try_insert(name, i).unwrap();
            i += 1;
        }
        for (name, _) in collection.switches.iter() {
            self.function_to_index.try_insert(name, i).unwrap();
            i += 1;
        }

        self.functions.push(Function::Block(FunctionBlock {
            instructions: Default::default(),
            next: Some(self.function(entry)?),
        }));

        for (_name, func) in collection.functions.iter() {
            let f = match func {
                crate::Function::User {
                    instructions: instrs,
                    next,
                } => {
                    let mut builder = FunctionBuilder {
                        program: self,
                        instructions: Default::default(),
                    };

                    for instr in instrs.iter() {
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
                    }

                    builder.finish(next)?
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
                    FunctionBlock {
                        instructions: [Instruction::Sys { id }].into(),
                        next: None,
                    }
                }
            };
            self.functions.push(Function::Block(f));
        }

        for (_name, switch) in collection.switches.iter() {
            let (register, ty) = self.register(&switch.register)?;
            match &self.types[ty.0 as usize] {
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
                    self.functions.push(Function::Switch(FunctionSwitch {
                        register,
                        cases: cases.into(),
                        default,
                    }));
                }
                Type::Fp32 => todo!(),
                Type::Opaque { .. } => todo!(),
                Type::Group { .. } => todo!(),
            }
        }

        Ok(())
    }

    fn register(&self, name: &'a Str) -> Result<(&RegisterMap<'a>, TypeId), Error> {
        self.register_to_index
            .get(name)
            .map(|(x, y)| (x, *y))
            .ok_or_else(|| Error {
                kind: ErrorKind::RegisterNotFound(name.to_string()),
                line: 0,
            })
    }

    fn array_register(&self, name: &'a Str) -> Result<(&RegisterMap<'a>, TypeId, TypeId), Error> {
        self.array_register_to_index
            .get(name)
            .map(|(x, y, z)| (x, *y, *z))
            .ok_or_else(|| Error {
                kind: ErrorKind::RegisterNotFound(name.to_string()),
                line: 0,
            })
    }

    fn function(&self, name: &Str) -> Result<u32, Error> {
        self.function_to_index
            .get(name)
            .ok_or_else(|| Error {
                kind: ErrorKind::FunctionNotFound(name.to_string()),
                line: 0,
            })
            .copied()
    }

    fn get_or_add_const(&mut self, ty: TypeId, value: &'a Str) -> Result<u32, Error> {
        let value = {
            match &self.types[ty.0 as usize] {
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
            .constant_to_index
            .entry(value.clone())
            .or_insert_with(|| {
                let i = u32::try_from(self.constants.len()).unwrap();
                self.constants.push(value);
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

    fn type_index_dimensions(&self, ty: TypeId) -> Option<Vec<u32>> {
        match &self.types[ty.0 as usize] {
            Type::Int(bits) => 1u32.checked_shl(u32::from(*bits)).map(|n| [n].into()),
            Type::Fp32 => None,
            Type::ConstantString => None,
            // Don't allow indexing on opaque since
            // - the bit length may change at any time
            // - opaque values may represent handles,
            //   and the value referenced may change without the opaque handle itself changing
            Type::Opaque { .. } => None,
            Type::Enum { value_to_id } => Some([u32::try_from(value_to_id.len()).unwrap()].into()),
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
                self.instructions.push(Instruction::Set { to, from });
            }
            RegisterMap::Group { fields } => {
                let csts = &self.program.group_constants[ty.0 as usize].unwrap();
                let csts = &csts[value];
                assert_eq!(fields.len(), csts.len());

                let Type::Group { fields: ty_fields } = &self.program.types[ty.0 as usize] else {
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
                instructions.push(Instruction::Move { to, from });
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
                instructions.push(Instruction::ArrayAccess { array });
                Self::array_recursive_index(instructions, index)?;
                instructions.push(if is_to {
                    Instruction::ArrayStore { register }
                } else {
                    Instruction::ArrayLoad { register }
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
            &RegisterMap::Unit { index } => instructions.push(Instruction::ArrayIndex { index }),
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
        self.instructions.push(Instruction::Call { address });
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
            .transpose()?;
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
