use {
    crate::{BuiltinType, Collection, Error, ErrorKind, Map, PrefixMap, Str},
    core::fmt,
    std::rc::Rc,
};

#[derive(Debug)]
pub struct Program {
    pub(crate) constants: Box<[Constant]>,
    pub(crate) registers: Box<[Register]>,
    pub(crate) array_registers: Box<[ArrayRegister]>,
    pub(crate) functions: Box<[Function]>,
    pub(crate) strings_buffer: Box<[u8]>,
    pub(crate) sys_to_registers: Box<[Box<[u32]>]>,
}

#[derive(Debug)]
pub struct Constant {
    pub value: Box<[u8]>,
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
    Group { fields: PrefixMap<&'a str, Self> },
}

#[derive(Debug)]
pub(crate) struct Function(pub Box<[Instruction]>);

#[derive(Debug, Default)]
struct ProgramBuilder<'a> {
    types_to_index: Map<&'a Str, TypeId>,
    types: Vec<Type<'a>>,
    /// type -> name -> values
    ///
    /// Group uses a PrefixMap, which is ordered.
    /// Box<[Str]> is hence also ordered
    group_constants: Vec<Map<&'a Str, Box<[&'a Str]>>>,
    function_to_index: Map<&'a Str, u32>,
    functions: Vec<Function>,
    constant_to_index: Map<Box<[u8]>, u32>,
    constants: Vec<Constant>,
    register_to_index: PrefixMap<&'a str, (RegisterMap<'a>, TypeId)>,
    registers: Vec<Register>,
    array_register_to_index: PrefixMap<&'a str, (RegisterMap<'a>, TypeId, TypeId)>,
    array_registers: Vec<ArrayRegister>,
    builtin_types: Map<BuiltinType, TypeId>,
    strings_buffer: Vec<u8>,
    sys_to_registers: Map<u32, Vec<u32>>,
}

#[derive(Debug)]
struct FunctionBuilder<'a, 'b> {
    program: &'b mut ProgramBuilder<'a>,
    instructions: Vec<Instruction>,
}

#[derive(Debug)]
enum Type<'a> {
    Integer32,
    Natural32,
    ConstantString,
    Enum { value_to_id: Map<&'a Str, u32> },
    Group { fields: PrefixMap<&'a str, TypeId> },
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
    /// Pop and return to popped address.
    Return,
    /// Jump to address.
    Jump { address: u32 },
    /// Jump if a register matches a constant.
    JumpEq {
        address: u32,
        register: u32,
        constant: u32,
    },
    /// Call a builtin function.
    SystemCall { id: u32 },
}

impl Program {
    pub fn from_collection(collection: &Collection, entry: &Str) -> Result<Self, Error> {
        let mut builder = ProgramBuilder::default();

        let stub_entry = "".to_string().into_boxed_str();

        builder.collect_types(collection)?;
        builder.collect_group_constants(collection)?;
        builder.collect_registers(collection)?;
        builder.collect_array_registers(collection)?;
        builder.collect_functions(collection, entry, &stub_entry)?;

        let len = builder
            .sys_to_registers
            .keys()
            .map(|&k| usize::try_from(k).unwrap())
            .max()
            .map_or(0, |x| x + 1);
        let mut sys_to_registers = (0..len).map(|_| Box::default()).collect::<Box<[_]>>();

        for (k, v) in builder.sys_to_registers {
            sys_to_registers[usize::try_from(k).unwrap()] = v.into();
        }

        Ok(Self {
            constants: builder.constants.into(),
            registers: builder.registers.into(),
            array_registers: builder.array_registers.into(),
            functions: builder.functions.into(),
            strings_buffer: builder.strings_buffer.into(),
            sys_to_registers,
        })
    }

    pub fn max_call_depth(&self, entry: u32) -> u32 {
        self.functions[entry as usize]
            .0
            .iter()
            .map(|i| match i {
                &Instruction::Call { address } => 1 + self.max_call_depth(address),
                &Instruction::Jump { address } | &Instruction::JumpEq { address, .. } => {
                    self.max_call_depth(address)
                }
                _ => 0,
            })
            .max()
            .unwrap_or(0)
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
            Self::Return => f.write_str("RETURN"),
            Self::Jump { address } => write!(f, "JUMP    {address}"),
            Self::JumpEq {
                address,
                register,
                constant,
            } => write!(f, "JUMPEQ  {address} {register} {constant}"),
            Self::SystemCall { id } => write!(f, "SYS     {id:?}"),
        }
    }
}

impl<'a> ProgramBuilder<'a> {
    fn collect_types(&mut self, collection: &'a Collection) -> Result<(), Error> {
        for (i, (name, ty)) in collection.types.iter().enumerate() {
            let i = u32::try_from(i).unwrap();
            self.types_to_index.try_insert(name, TypeId(i)).unwrap();
        }

        for (name, ty) in collection.types.iter() {
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
                        .map(|(name, ty)| (&**name, self.types_to_index[ty]))
                        .collect();
                    Type::Group { fields }
                }
                &crate::Type::Builtin(b) => {
                    let i = self.types_to_index[name];
                    self.builtin_types.try_insert(b, i).unwrap();
                    match b {
                        BuiltinType::Integer32 => Type::Integer32,
                        BuiltinType::Natural32 => Type::Natural32,
                        BuiltinType::ConstantString => Type::ConstantString,
                    }
                }
            };
            self.types.push(ty);
        }
        Ok(())
    }

    fn collect_group_constants(&mut self, collection: &'a Collection) -> Result<(), Error> {
        self.group_constants
            .resize_with(self.types.len(), Default::default);

        for (ty, constants) in collection.constants.iter() {
            for (name, values) in constants.iter() {
                let ty = self.types_to_index[ty];
                let Type::Group { fields } = &self.types[ty.0 as usize] else {
                    todo!();
                };
                assert_eq!(fields.len(), values.len());
                let values = fields.iter().map(|(k, _)| &values[*k]).collect();
                self.group_constants[ty.0 as usize]
                    .try_insert(name, values)
                    .unwrap();
            }
        }

        Ok(())
    }

    fn collect_registers(&mut self, collection: &'a Collection) -> Result<(), Error> {
        for (name, ty) in collection.registers.iter() {
            let ty = self.types_to_index[ty];
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
                        stack.push((PrefixMap::default(), it, name));
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
                        stack.push((PrefixMap::default(), it, name));
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

        self.functions.push(Function(
            [Instruction::Jump {
                address: self.function(entry)?,
            }]
            .into(),
        ));

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
                        }
                    }

                    Function(builder.finish(next)?.into())
                }
                crate::Function::Builtin { id, registers } => {
                    let id = *id;
                    let regs = registers
                        .iter()
                        .map(|r| {
                            let (reg, ty) = self.register(&r.1)?;
                            match reg {
                                RegisterMap::Unit { index } => Ok(*index),
                                RegisterMap::Group { fields } => todo!(),
                            }
                        })
                        .try_collect()?;
                    self.sys_to_registers.try_insert(id, regs).unwrap();
                    Function([Instruction::SystemCall { id }, Instruction::Return].into())
                }
            };
            self.functions.push(f);
        }

        for (_name, func) in collection.switches.iter() {
            let mut instructions = Vec::new();
            let (register, ty) = self.register(&func.register)?;
            let ty = &self.types[ty.0 as usize];
            match ty {
                Type::Integer32 => todo!(),
                Type::Natural32 => todo!(),
                Type::ConstantString => todo!(),
                Type::Enum { value_to_id } => {
                    let all = value_to_id
                        .keys()
                        .all(|k| func.branches.iter().any(|(kk, _)| kk == &**k));
                    if !all && func.default.is_none() {
                        todo!()
                    }
                    for (i, (value, next)) in func.branches.iter().enumerate() {
                        let address = self.function(next)?;
                        let constant = *value_to_id.get(value).ok_or_else(|| todo!())?;
                        if func.default.is_some() || i < func.branches.len() - 1 {
                            match register {
                                &RegisterMap::Unit { index: register } => {
                                    instructions.push(Instruction::JumpEq {
                                        address,
                                        register,
                                        constant,
                                    });
                                }
                                RegisterMap::Group { fields } => todo!(),
                            }
                        } else {
                            instructions.push(Instruction::Jump { address });
                        }
                    }
                }
                Type::Group { .. } => todo!(),
            }
            self.functions.push(Function(instructions.into()));
        }

        Ok(())
    }

    fn ty(&self, name: &Str) -> Result<TypeId, Error> {
        self.types_to_index
            .get(name)
            .ok_or_else(|| Error {
                kind: ErrorKind::TypeNotFound(name.to_string()),
            })
            .copied()
    }

    fn register(&self, name: &'a Str) -> Result<(&RegisterMap<'a>, TypeId), Error> {
        let (mut reg, mut ty, mut post) = self
            .register_to_index
            .get(name)
            .map(|((x, y), z)| (x, *y, z))
            .ok_or_else(|| Error {
                kind: ErrorKind::RegisterNotFound(name.to_string()),
            })?;

        while post != "" {
            let RegisterMap::Group { fields } = reg else {
                unreachable!()
            };
            let tyty = &self.types[usize::try_from(ty.0).unwrap()];
            let Type::Group { fields: ty_fields } = tyty else {
                unreachable!()
            };
            ty = *ty_fields.get(post).unwrap().0;
            (reg, post) = fields.get(post).unwrap();
        }

        Ok((reg, ty))
    }

    fn array_register(&self, name: &'a Str) -> Result<(&RegisterMap<'a>, TypeId, TypeId), Error> {
        let (mut reg, index_ty, mut value_ty, mut post) = self
            .array_register_to_index
            .get(name)
            .map(|((x, y, z), w)| (x, *y, *z, w))
            .ok_or_else(|| Error {
                kind: ErrorKind::RegisterNotFound(name.to_string()),
            })?;

        while post != "" {
            let RegisterMap::Group { fields } = reg else {
                unreachable!()
            };
            let tyty = &self.types[usize::try_from(value_ty.0).unwrap()];
            let Type::Group { fields: ty_fields } = tyty else {
                unreachable!()
            };
            value_ty = *ty_fields.get(post).unwrap().0;
            (reg, post) = fields.get(post).unwrap();
        }

        Ok((reg, index_ty, value_ty))
    }

    fn function(&self, name: &Str) -> Result<u32, Error> {
        self.function_to_index
            .get(name)
            .ok_or_else(|| Error {
                kind: ErrorKind::FunctionNotFound(name.to_string()),
            })
            .copied()
    }

    fn get_or_add_const(&mut self, ty: TypeId, value: &'a Str) -> Result<u32, Error> {
        let value = {
            match &self.types[ty.0 as usize] {
                Type::Integer32 => self.parse_integer32(value)?.to_ne_bytes().to_vec(),
                Type::Natural32 => self.parse_natural32(value)?.to_ne_bytes().to_vec(),
                Type::ConstantString => {
                    let (offt, len) = self.parse_constant_string(value)?;
                    let mut b = [0; 8];
                    b[..4].copy_from_slice(&offt.to_ne_bytes());
                    b[4..].copy_from_slice(&len.to_ne_bytes());
                    b.to_vec()
                }
                Type::Enum { value_to_id } => {
                    let id = value_to_id[value];
                    if value_to_id.len() < 1 << 8 {
                        (id as u8).to_ne_bytes().to_vec()
                    } else if value_to_id.len() < 1 << 16 {
                        (id as u16).to_ne_bytes().to_vec()
                    } else if value_to_id.len() < 1 << 32 {
                        (id as u32).to_ne_bytes().to_vec()
                    } else {
                        (id as u64).to_ne_bytes().to_vec()
                    }
                }
                Type::Group { .. } => todo!("group constants?"),
            }
        };

        let value = value.into_boxed_slice();

        let i = self
            .constant_to_index
            .entry(value.clone())
            .or_insert_with(|| {
                let i = u32::try_from(self.constants.len()).unwrap();
                self.constants.push(Constant { value });
                i
            });

        Ok(*i)
    }

    fn parse_constant_string(&mut self, value: &Str) -> Result<(u32, u32), Error> {
        let offt = u32::try_from(self.strings_buffer.len()).unwrap();

        let mut it = value.bytes();
        if it.next() != Some(b'"') {
            todo!();
        }

        loop {
            let c = it.next().unwrap();
            match c {
                b'\\' => todo!(),
                b'"' => break,
                c => self.strings_buffer.push(c),
            }
        }

        if it.next().is_some() {
            todo!();
        }

        let len = u32::try_from(self.strings_buffer.len()).unwrap() - offt;

        Ok((offt, len))
    }

    fn parse_natural32(&mut self, value: &Str) -> Result<u32, Error> {
        self.parse_natural128(value)
            .and_then(|n| Ok(n.try_into().unwrap()))
    }

    fn parse_integer32(&mut self, value: &Str) -> Result<i32, Error> {
        let (neg, s) = if value.starts_with("-") {
            (true, &value[1..])
        } else {
            (false, &value[..])
        };
        let n = self.parse_natural128(s)?;
        let n = i128::try_from(n).unwrap();
        let n = i32::try_from(if neg { -n } else { n }).unwrap();
        Ok(n)
    }

    fn parse_char(&mut self, value: &str) -> Result<u32, Error> {
        let mut it = value.chars();
        if it.next() != Some('\'') {
            todo!();
        }

        let c = match it.next() {
            Some('\\') => todo!(),
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
                _ => todo!(),
            };
            let x = u128::from(x);
            if x >= base {
                todo!();
            }
            n = n.checked_mul(base).ok_or_else(|| todo!())?;
            n = n.checked_add(x).ok_or_else(|| todo!())?;
        }

        Ok(n)
    }

    fn type_index_dimensions(&self, ty: TypeId) -> Option<Vec<u32>> {
        match &self.types[ty.0 as usize] {
            Type::Integer32 | Type::Natural32 | Type::ConstantString => None,
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
                let csts = &self.program.group_constants[ty.0 as usize];
                let csts = &csts[value];
                assert_eq!(fields.len(), csts.len());

                let Type::Group { fields: ty_fields } = &self.program.types[ty.0 as usize] else {
                    todo!()
                };

                //FIXME
                let csts = csts.clone();
                let ty_fields = ty_fields.clone();

                for (((_, r), (_, tyty)), c) in fields.iter().zip(ty_fields.iter()).zip(csts.iter())
                {
                    self.set_recursive(r, *tyty, *c)?;
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

        fn f(instructions: &mut Vec<Instruction>, to: &RegisterMap<'_>, from: &RegisterMap<'_>) {
            use RegisterMap::*;
            match (to, from) {
                (&Unit { index: to }, &Unit { index: from }) => {
                    instructions.push(Instruction::Move { to, from });
                }
                (Group { fields: to }, Group { fields: from }) => {
                    for ((k1, to), (k2, from)) in to.iter().zip(from.iter()) {
                        debug_assert_eq!(k1, k2);
                        f(instructions, to, from);
                    }
                }
                _ => todo!(),
            }
        }

        f(&mut self.instructions, to, from);
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

    fn finish(mut self, next: &'a Option<Str>) -> Result<Vec<Instruction>, Error> {
        if let Some(f) = next.as_ref() {
            let address = self.program.function(f)?;
            self.instructions.push(Instruction::Jump { address })
        } else {
            self.instructions.push(Instruction::Return)
        }
        Ok(self.instructions)
    }
}

impl<'a> Type<'a> {
    /// Minimum size in bits.
    pub fn bit_size(&self) -> u32 {
        match self {
            Self::Integer32 | Self::Natural32 => 32,
            // offset + length
            Self::ConstantString => 32 + 32,
            Self::Enum { value_to_id } => value_to_id
                .len()
                .checked_next_power_of_two()
                .map_or(usize::BITS, |x| x.try_into().unwrap()),
            Self::Group { fields } => todo!(),
        }
    }

    /// Minimum size in bytes.
    pub fn byte_size(&self) -> u32 {
        (self.bit_size() + 7) / 8
    }
}
