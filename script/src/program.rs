use crate::{function::{ArgMode, Statement, Value}, Collection, Map, Record, Str};

#[derive(Debug)]
pub struct Program {
    functions: Box<[Box<[Function]>]>,
    constants: Box<[Box<[u8]>]>,
}

#[derive(Debug)]
pub struct Error;

#[derive(Debug)]
pub struct Debug<'a> {
    type_names: Vec<&'a Str>,
    function_names: Vec<&'a Str>,
}

#[derive(Debug)]
struct ProgramBuilder<'a> {
    collection: &'a Collection,
    types_to_index: Map<&'a Str, u32>,
    types: Vec<Type<'a>>,
    function_to_index: Map<&'a Str, u32>,
    functions: Vec<Box<[Function]>>,
    function_var_to_register: Vec<Box<[Map<&'a Str, u32>]>>,
    constants: Vec<Box<[u8]>>,
}

#[derive(Debug)]
struct Type<'a> {
    len: u32,
    fields: Map<&'a Str, Field>,
}

#[derive(Debug)]
struct Field {
    offset: u32,
    format: FieldFormat,
}

#[derive(Debug)]
enum FieldFormat {
    One { ty: TypeId },
    Tag { types: Box<[TypeId]> },
}

#[derive(Debug)]
struct Function {
    args: Box<[RegisterIndex]>,
    registers: Box<[TypeId]>,
    instructions: Box<[Call]>,
    lambdas: Box<[Box<[Call]>]>,
}

#[derive(Debug)]
struct Call {
    function: FunctionIndex,
    move_in: Box<[MoveIn]>,
    move_out: Box<[MoveOut]>,
}

#[derive(Debug)]
struct MoveIn {
    src: Box<[RegisterIndex]>,
    dst: RegisterIndex,
}

#[derive(Debug)]
struct MoveOut {
    src: RegisterIndex,
    dst: Box<[RegisterIndex]>,
}

#[derive(Debug)]
struct FunctionIndex(u32);

#[derive(Debug)]
struct RegisterIndex(u32);

#[derive(Debug)]
struct TypeId(u32);

impl Program {
    pub fn from_collection(collection: &Collection) -> Result<Self, Error> {
        let mut builder = ProgramBuilder {
            collection,
            types_to_index: Default::default(),
            types: Default::default(),
            function_to_index: Default::default(),
            functions: Default::default(),
            function_var_to_register: Default::default(),
            constants: Default::default(),
        };

        builder.collect_types()?;
        builder.collect_functions()?;

        Ok(Self {
            functions: builder.functions.into(),
            constants: builder.constants.into(),
        })
    }
}

impl<'a> ProgramBuilder<'a> {
    /// Collect types.
    fn collect_types(&mut self) -> Result<(), Error> {
        for (name, record) in self.collection.data.iter() {
            let ty = match record {
                Record::User { fields } => Type {
                    len: u32::MAX,
                    fields: Default::default(),
                },
                Record::Opaque { size } => Type {
                    len: *size,
                    fields: Default::default(),
                }
            };
        }

        for name in self.collection.data.keys() {
            let _ = self.resolve_type(name)?;
        }

        Ok(())
    }

    /// Resolve type length
    fn resolve_type(&mut self, name: &'a Str) -> Result<u32, Error> {
        let index = self.types_to_index[name];

        let f = |s: &mut Self| &mut s.types[index as usize];

        let len = f(self).len;
        if len != u32::MAX {
            return Ok(len);
        }

        let Record::User { fields } = &self.collection.data[name] else { unreachable!() };

        let mut len = 0;
        let mut n_fields = Map::new();
        for (name, types) in fields.iter() {
            assert!(!types.is_empty());
            let mut field_len = 0;
            for ty in types.iter() {
                field_len = field_len.max(self.resolve_type(ty)?);
            }
            let f = |n| self.types_to_index[n];
            let format = if types.len() == 1 {
                FieldFormat::One { ty: f(&types[0]) }
            } else {
                field_len += 4;
                FieldFormat::Tag { types: types.iter().map(f).collect() }
            };
            n_fields.insert(name, Field {
                offset: len,
                format,
            });
            len += field_len;
        }

        let ty = f(self);
        ty.len = len;
        ty.fields = n_fields;

        Ok(len)
    }

    /// Collect functions.
    fn collect_functions(&mut self) -> Result<(), Error> {
        for (name, list) in self.collection.functions.iter() {
            let mut n_list = Vec::new();
            let mut n_var_to_register = Vec::new();
            for f in list.iter() {
                let mut args = Vec::new();
                let mut var_to_register = Map::new();
                let mut register_offset = 0;
                for ((mode, ty), name) in f.args.iter().zip(f.arg_names.iter()) {
                    var_to_register.insert(name, register_offset);
                    register_offset += 0;
                }
                for stmt in f.statements.iter() {
                    match stmt {
                        Statement::Call { function, args } => todo!(),
                        Statement::Variable { name } => todo!(),
                        Statement::Constant { name } => todo!(),
                    }
                }
                n_list.push(Function {
                    args,
                    instructions: Vec::new(),
                });
                n_var_to_register.push(var_to_register);
            }
            self.function_to_index.insert(name, self.functions.len().try_into().unwrap());
            self.functions.push(n_list.into());
            self.function_var_to_register.push(n_var_to_register.into())
        }

        self.parse_functions()?;

        Ok(())
    }

    fn parse_functions(&mut self) -> Result<(), Error> {
        for (name, ff_l) in self.collection.functions.iter() {
            let i = self.function_to_index[name];
            let f_l = &self.functions[i as usize];
            let v2i_l = &self.function_var_to_register[i as usize];
            for (ff, (f, v2i)) in ff_l.iter().zip(f_l.iter_mut().zip(v2i_l.iter())) {
                assert!(f.instructions.is_empty());
                for stmt in ff.statements.iter() {
                    let Statement::Call { function, args } = stmt else { continue };
                    let f_index = self.function_to_index[function];
                    f.instructions.push(Instruction::SetTarget { function: f_index, index: u32::MAX });
                    for (mode, arg) in args.iter() {
                        match mode {
                            ArgMode::In | ArgMode::Ref => {
                                f.instructions.push(Instruction::MoveTo { from, to: () });
                            }
                            ArgMode::Out => {}
                        }
                    }
                    f.instructions.push(Instruction::Call);
                    for (mode, arg) in args.iter() {
                        match mode {
                            ArgMode::Out | ArgMode::Ref => {
                                f.instructions.push(Instruction::MoveFrom { from: (), to: () });
                            }
                            ArgMode::In => {}
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn parse_functions_one(&mut self, name: &'a Str, f: &'a crate::Function) -> Result<(), Error> {
        let mut vars = Map::new();

        {
            let label = FunctionLabel::<'a> {
                name,
                args: Cow::from(&*f.args),
            };
            let v = self.function_map.entry(label).or_default();
            if v.location.is_some() {
                return Err(Error);
            }
            v.location = Some(self.bytecode.len().try_into().unwrap());
        }

        for (name, (mode, ty)) in f.arg_names.iter().zip(f.args.iter()) {
            let index = vars.len().try_into().unwrap();
            let (status, mode) = match mode {
                ArgMode::In => (VarStatus::Init, VarMode::In { index }),
                ArgMode::Out => (VarStatus::Uninit, VarMode::Out { index }),
                ArgMode::Ref => (VarStatus::Init, VarMode::Ref { index }),
            };
            if vars.try_insert(name, Var { status, mode, ty }).is_err() {
                return Err(Error);
            }
        }

        let i = ();

        for (i, stmt) in f.statements.iter().enumerate() {
            match stmt {
                Statement::Variable { name } | Statement::Constant { name } => {
                    dbg!(name);
                    todo!();
                }
                Statement::Call { function, args } => {
                    dbg!(function, args, &vars);
                    let mut arg_types = Vec::new();
                    for (i, (mode, value)) in args.iter().enumerate() {
                        let i = u32::try_from(i).unwrap();
                        let ty = match value {
                            Value::Register(name) => {
                                let var = vars.get(name).ok_or(Error)?;
                                match mode {
                                    ArgMode::In => todo!(),
                                    ArgMode::Out => todo!(),
                                    ArgMode::Ref => {
                                        if !var.is_init() {
                                            return Err(Error);
                                        }
                                        self.set_arg(var, &[], i);
                                    }
                                }
                                var.ty.clone()
                            }
                            Value::String(s) => {
                                if mode == &ArgMode::Out {
                                    return Err(Error);
                                }
                                let val = self.add_slice(s.as_bytes());
                                self.op_setarg_const(val, i);
                                "ConstantString".into()
                            }
                        };
                        arg_types.push((*mode, ty));
                    }
                    self.op_call(FunctionLabel {
                        name: function,
                        args: Cow::from(arg_types),
                    }, )
                }
            }
        }

        Ok(())
    }

    fn add_slice(&mut self, data: &[u8]) -> u32 {
        let i = self.constants.len().try_into().unwrap();
        self.constants.extend(u32::try_from(data.len()).unwrap().to_ne_bytes());
        self.constants.extend_from_slice(data);
        i
    }
}

impl<'a> Var<'a> {
    fn is_init(&self) -> bool {
        matches!(&self.status, VarStatus::Init)
    }
}
