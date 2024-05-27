use crate::{BuiltinFunction, BuiltinRegister, BuiltinType, Collection, Error, Map, Str};

#[derive(Debug)]
pub struct Program {
    pub(crate) constants: Box<[Constant]>,
    pub(crate) registers: Box<[Register]>,
    pub(crate) functions: Box<[Function]>,
    pub(crate) strings_buffer: Box<[u8]>,
    pub(crate) entry: u32,
}

#[derive(Debug)]
pub struct Constant {
    pub value: Box<[u8]>,
}

#[derive(Debug)]
pub struct Register {
    pub bits: u32,
    pub builtin: Option<BuiltinRegister>,
}

#[derive(Debug)]
pub(crate) struct Function(pub Box<[Instruction]>);

#[derive(Debug, Default)]
struct ProgramBuilder<'a> {
    types_to_index: Map<&'a Str, u32>,
    types: Vec<Type<'a>>,
    function_to_index: Map<&'a Str, u32>,
    functions: Vec<Function>,
    constant_to_index: Map<Box<[u8]>, u32>,
    constants: Vec<Constant>,
    register_to_index: Map<&'a Str, (u32, TypeId)>,
    registers: Vec<Register>,
    builtin_registers: Map<BuiltinRegister, u32>,
    builtin_types: Map<BuiltinType, u32>,
    strings_buffer: Vec<u8>,
}

#[derive(Debug)]
enum Type<'a> {
    Integer32,
    Natural32,
    ConstantString,
    User { value_to_id: Map<&'a Str, u32> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TypeId(u32);

#[derive(Debug)]
pub(crate) enum Instruction {
    /// Copy one register to another.
    Move { to: u32, from: u32 },
    /// Set register value based on value in constants map.
    Set { to: u32, from: u32 },
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
    SystemCall { function: BuiltinFunction },
}

impl Program {
    pub fn from_collection(collection: &Collection, entry: &Str) -> Result<Self, Error> {
        let mut builder = ProgramBuilder::default();

        builder.collect_types(collection)?;
        builder.collect_registers(collection)?;
        builder.collect_functions(collection)?;

        Ok(Self {
            constants: builder.constants.into(),
            registers: builder.registers.into(),
            functions: builder.functions.into(),
            strings_buffer: builder.strings_buffer.into(),
            entry: builder.function_to_index[entry],
        })
    }

    pub fn max_call_depth(&self, entry: u32) -> u32 {
        self.functions[entry as usize]
            .0
            .iter()
            .map(|i| match i {
                &Instruction::Call { address } => 1 + self.max_call_depth(address),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }
}

impl<'a> ProgramBuilder<'a> {
    fn collect_types(&mut self, collection: &'a Collection) -> Result<(), Error> {
        for (name, ty) in collection.types.iter() {
            let i = u32::try_from(self.types.len()).unwrap();
            self.types_to_index.try_insert(name, i).unwrap();
            let ty = match ty {
                crate::Type::User(values) => {
                    let value_to_id = values
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (v, u32::try_from(i).unwrap()))
                        .collect();
                    Type::User { value_to_id }
                }
                &crate::Type::Builtin(b) => {
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

    fn collect_registers(&mut self, collection: &'a Collection) -> Result<(), Error> {
        for (name, reg) in collection.registers.iter() {
            dbg!(name, reg);
            let i = u32::try_from(self.registers.len()).unwrap();
            let (ty, builtin) = match reg {
                crate::Register::User(ty) => (self.types_to_index[ty], None),
                &crate::Register::Builtin(which) => {
                    self.builtin_registers.try_insert(which, i).unwrap();
                    let ty = match which {
                        BuiltinRegister::WriteConstantStringValue => BuiltinType::ConstantString,
                    };
                    (self.builtin_types[&ty], Some(which))
                }
            };
            self.register_to_index.insert(name, (i, TypeId(ty)));
            let bits = self.types[ty as usize].bit_size();
            self.registers.push(Register { bits, builtin });
        }
        Ok(())
    }

    fn collect_functions(&mut self, collection: &'a Collection) -> Result<(), Error> {
        // prepass so we can immediately resolve addresses
        let mut i = 0;
        for (name, _) in collection.functions.iter() {
            dbg!(name);
            self.function_to_index.try_insert(name, i).unwrap();
            i += 1;
        }

        dbg!(&self.function_to_index);

        for (_name, func) in collection.functions.iter() {
            dbg!(_name);
            let f = match func {
                crate::Function::User {
                    instructions: instrs,
                    next,
                } => {
                    let mut instructions = Vec::new();

                    for instr in instrs.iter() {
                        dbg!(instr);
                        match instr {
                            crate::Instruction::Set { to, value } => {
                                let (to, ty) = self.register_to_index[to];
                                let from = self.get_or_add_const(ty, value)?;
                                instructions.push(Instruction::Set { to, from });
                            }
                            crate::Instruction::Move { to, from } => {
                                let to = self.register_to_index[to].0;
                                let from = self.register_to_index[from].0;
                                instructions.push(Instruction::Move { to, from });
                            }
                            crate::Instruction::Call { function } => {
                                let address = self.function_to_index[function];
                                instructions.push(Instruction::Call { address });
                            }
                        }
                    }

                    if let Some(f) = next.as_ref() {
                        dbg!(f);
                        let address = self.function_to_index[f];
                        instructions.push(Instruction::Jump { address })
                    } else {
                        instructions.push(Instruction::Return)
                    }

                    Function(instructions.into())
                }
                &crate::Function::Builtin(function) => {
                    Function([Instruction::SystemCall { function }, Instruction::Return].into())
                }
            };
            self.functions.push(f);
        }

        Ok(())
    }

    fn get_or_add_const(&mut self, ty: TypeId, value: &Str) -> Result<u32, Error> {
        let value = {
            match &self.types[ty.0 as usize] {
                Type::Integer32 => value.parse::<i32>().unwrap().to_ne_bytes().to_vec(),
                Type::Natural32 => value.parse::<u32>().unwrap().to_ne_bytes().to_vec(),
                Type::ConstantString => {
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
                    let mut b = [0; 8];
                    b[..4].copy_from_slice(&offt.to_ne_bytes());
                    b[4..].copy_from_slice(&len.to_ne_bytes());
                    b.to_vec()
                }
                Type::User { value_to_id } => {
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
}

impl<'a> Type<'a> {
    /// Minimum size in bits.
    pub fn bit_size(&self) -> u32 {
        match self {
            Self::Integer32 | Self::Natural32 => 32,
            // offset + length
            Self::ConstantString => 32 + 32,
            Self::User { value_to_id } => value_to_id
                .len()
                .checked_next_power_of_two()
                .map_or(usize::BITS, |x| x.try_into().unwrap()),
        }
    }

    /// Minimum size in bytes.
    pub fn byte_size(&self) -> u32 {
        (self.bit_size() + 7) / 8
    }
}
