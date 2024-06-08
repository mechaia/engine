use {
    crate::{
        program::{Constant, Function, FunctionBlock, FunctionSwitch, Instruction, Register},
        Map, Program,
    },
    core::fmt,
};

pub struct WordVM {
    instructions: Box<[u32]>,
    registers_size: u32,
    stack_size: u32,
    strings_offset: u32,
    sys_to_registers: Box<[Box<[u32]>]>,
}

#[derive(Debug)]
pub struct WordVMState {
    registers: Box<[u32]>,
    stack: Box<[u32]>,
    stack_index: u32,
    instruction_index: u32,
    array_index: u32,
    compare_state: u32,
}

#[derive(Clone, Debug)]
pub enum Yield {
    Finish,
    Sys { id: u32 },
    Preempt,
}

#[derive(Clone, Debug)]
pub struct Error;

#[derive(Default)]
struct OpEncoder {
    instructions: Vec<u32>,
    label_to_address: Map<u32, u32>,
    resolve_label: Map<u32, (Resolve, u32)>,
    resolve_constant: Map<u32, ()>,
}

#[derive(Debug)]
enum Resolve {
    Full,
    Shift4,
}

const OP4_MOVE: u32 = 0;
const OP4_SET: u32 = 1;
const OP4_COND_EQ: u32 = 2;
const OP4_COND_ANDEQ: u32 = 3;
const OP4_COND_JUMP: u32 = 4;
const OP4_JUMP: u32 = 5;
const OP4_CALL: u32 = 6;
const OP4_RET: u32 = 7;
const OP4_SYS: u32 = 8;

/// index = base
const OP4_ARRAY_SET: u32 = 9;
/// index += stride * [register]
const OP4_ARRAY_ADD: u32 = 10;
/// [register] = [index]
/// index += 1
const OP4_ARRAY_LOAD: u32 = 11;
/// [index] = [register]
/// index += 1
const OP4_ARRAY_STORE: u32 = 12;

const OP4_UNDEF: u32 = 15;

impl WordVM {
    pub fn from_program(program: &Program) -> Self {
        let mut encoder = OpEncoder::default();

        let mut registers_size = 0;
        let mut register_offsets = Vec::new();
        let mut array_register_offsets = Vec::new();

        for reg in program.registers.iter() {
            register_offsets.push(registers_size);
            registers_size += words_for_bits(reg.bits);
        }

        for reg in program.array_registers.iter() {
            array_register_offsets.push(registers_size);
            registers_size += words_for_bits(reg.bits) * reg.dimensions.iter().product::<u32>();
        }

        let sys_to_registers = program
            .sys_to_registers
            .iter()
            .map(|v| {
                v.iter()
                    .map(|i| register_offsets[usize::try_from(*i).unwrap()])
                    .collect()
            })
            .collect();

        let mut conv = |i, f: &Function| {
            encoder
                .label_to_address
                .try_insert(i, encoder.cur())
                .unwrap();
            match f {
                Function::Block(b) => {
                    let mut instructions = b.instructions.iter();
                    while let Some(instr) = instructions.next() {
                        let r = |i: u32| register_offsets[i as usize];
                        let l = |i: u32| words_for_bits(program.registers[i as usize].bits);
                        match instr {
                            &Instruction::Move { to, from } => {
                                for i in 0..l(to) {
                                    encoder.op_move(r(from) + i, r(to) + i)
                                }
                            }
                            &Instruction::Set { to, from } => {
                                let constant = &program.constants[from as usize];
                                for (i, w) in constant.value.chunks(4).enumerate() {
                                    let mut ww = [0; 4];
                                    ww[..w.len()].copy_from_slice(w);
                                    let value = u32::from_ne_bytes(ww);
                                    encoder.op_set(value, r(to) + i as u32);
                                }
                            }
                            &Instruction::ArrayAccess { array } => {
                                encoder.op_array_set(array_register_offsets[array as usize]);
                                let array = &program.array_registers[array as usize];
                                let element_size = words_for_bits(array.bits);
                                let mut stride =
                                    element_size * array.dimensions.iter().product::<u32>();
                                for i in 0.. {
                                    match instructions.next().unwrap() {
                                        &Instruction::ArrayIndex { index } => {
                                            stride /= array.dimensions[i];
                                            encoder.op_array_add(stride, r(index));
                                        }
                                        &Instruction::ArrayLoad { register } => {
                                            encoder.op_array_load(r(register));
                                            break;
                                        }
                                        &Instruction::ArrayStore { register } => {
                                            encoder.op_array_store(r(register));
                                            break;
                                        }
                                        _ => todo!(),
                                    }
                                }
                            }
                            Instruction::ArrayIndex { .. }
                            | Instruction::ArrayLoad { .. }
                            | Instruction::ArrayStore { .. } => todo!(),
                            &Instruction::Call { address } => encoder.op_call(address),
                            &Instruction::Sys { id } => encoder.op_sys(id),
                        }
                    }
                    if let Some(next) = b.next {
                        // elide redundant jump
                        if next != i + 1 {
                            encoder.op_jump(next);
                        }
                    } else {
                        encoder.op_ret();
                    }
                }
                Function::Switch(s) => {
                    let register = register_offsets[s.register as usize];
                    for case in s.cases.iter() {
                        let cst = &program.constants[case.constant as usize];
                        for (i, w) in cst.value.chunks(4).enumerate() {
                            let mut ww = [0; 4];
                            // FIXME endianness
                            ww[..w.len()].copy_from_slice(w);
                            let v = u32::from_ne_bytes(ww);
                            if i == 0 {
                                encoder.op_cond_eq(register, v)
                            } else {
                                encoder.op_cond_andeq(register, v)
                            }
                        }
                        encoder.op_cond_jump(case.function);
                    }
                    if let Some(default) = s.default {
                        // elide redundant jump
                        if default != i + 1 {
                            encoder.op_jump(default);
                        }
                    } else {
                        encoder.instructions.push(OP4_UNDEF);
                    }
                }
            }
        };

        for (i, f) in program.functions.iter().enumerate() {
            let i = u32::try_from(i).unwrap();
            conv(i, f);
        }

        let (instructions, strings_offset) = encoder.finish(&program.strings_buffer);
        let stack_size = program.max_call_depth(0);

        Self {
            instructions,
            registers_size,
            stack_size,
            strings_offset,
            sys_to_registers,
        }
    }

    pub fn create_state(&self) -> WordVMState {
        WordVMState {
            registers: vec![0; self.registers_size as usize].into(),
            stack: vec![0; self.stack_size as usize].into(),
            stack_index: 0,
            instruction_index: 0,
            array_index: 0,
            compare_state: 0,
        }
    }

    pub fn sys_registers(&self, id: u32) -> &[u32] {
        self.sys_to_registers
            .get(usize::try_from(id).unwrap())
            .map_or(&[], |v| &**v)
    }

    pub fn strings_buffer(&self) -> &[u8] {
        slice_u32_as_u8(&self.instructions[self.strings_offset as usize..])
    }
}

impl fmt::Debug for WordVM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stack size: {}\n", self.stack_size)?;
        write!(f, "registers size: {}\n", self.registers_size)?;
        write!(f, "strings offset: {}\n", self.strings_offset)?;
        f.write_str("instructions:\n")?;
        let mut i = 0;
        loop {
            let Some(op) = self.instructions[..self.strings_offset as usize].get(i) else {
                break;
            };
            write!(f, "{i:>7}: ")?;
            i += 1;
            let mut next_u32 = || {
                i += 1;
                self.instructions.get(i - 1).copied()
            };
            match *op & 15 {
                OP4_MOVE => {
                    let to = op >> 4;
                    match next_u32() {
                        Some(from) => write!(f, "MOVE    {to} {from}")?,
                        _ => write!(f, "MOVE    {to} ???")?,
                    }
                }
                OP4_SET => {
                    let to = op >> 4;
                    match next_u32() {
                        Some(value) => write!(f, "SET     {to} {value}")?,
                        _ => write!(f, "SET     {to} ???")?,
                    }
                }
                OP4_SYS => {
                    let id = op >> 4;
                    write!(f, "SYS     {id}")?
                }
                OP4_COND_EQ => {
                    let register = op >> 4;
                    if let Some(value) = next_u32() {
                        write!(f, "C.EQ    {register} {value}")?
                    } else {
                        write!(f, "C.EQ    {register} ???")?
                    }
                }
                OP4_COND_ANDEQ => {
                    let register = op >> 4;
                    if let Some(value) = next_u32() {
                        write!(f, "C.ANDEQ {register} {value}")?
                    } else {
                        write!(f, "C.ANDEQ {register} ???")?
                    }
                }
                OP4_COND_JUMP => {
                    let address = op >> 4;
                    write!(f, "C.JUMP  {address}")?
                }
                OP4_JUMP => {
                    let address = op >> 4;
                    write!(f, "JUMP    {address}\n")?
                }
                OP4_CALL => {
                    let address = op >> 4;
                    write!(f, "CALL    {address}")?
                }
                OP4_RET => f.write_str("RET\n")?,
                OP4_ARRAY_SET => write!(f, "A.SET   {}", op >> 4)?,
                OP4_ARRAY_ADD => {
                    if let Some(register) = next_u32() {
                        write!(f, "A.ADD   {} {}", op >> 4, register)?
                    } else {
                        write!(f, "A.ADD   {} ???", op >> 4)?
                    }
                }
                OP4_ARRAY_LOAD => write!(f, "A.LOAD  {}", op >> 4)?,
                OP4_ARRAY_STORE => write!(f, "A.STORE {}", op >> 4)?,
                OP4_UNDEF => f.write_str("UNDEF\n")?,
                op => write!(f, "??? {op}")?,
            }
            f.write_str("\n")?;
        }

        f.write_str("strings buffer: ")?;
        for c in slice_u32_as_u8(&self.instructions[self.strings_offset as usize..]) {
            // FIXME bruh
            write!(f, "{}", *c as char)?;
        }
        f.write_str("\n")?;

        f.write_str("sys register map:\n")?;
        for (id, regs) in self.sys_to_registers.iter().enumerate() {
            write!(f, "{id:>7}: {:?}\n", &regs)?;
        }

        Ok(())
    }
}

impl WordVMState {
    pub fn step(&mut self, vm: &WordVM) -> Result<Yield, Error> {
        let op = *vm
            .instructions
            .get(self.instruction_index as usize)
            .ok_or(Error)?;
        self.instruction_index += 1;
        let mut next_u32 = || {
            self.instruction_index += 1;
            vm.instructions
                .get(self.instruction_index as usize - 1)
                .copied()
                .ok_or(Error)
        };
        match op & 15 {
            OP4_MOVE => {
                let to = op >> 4;
                let from = next_u32()?;
                let x = self.register(from)?;
                self.set_register(to, x)?;
            }
            OP4_SET => {
                let to = op >> 4;
                let value = next_u32()?;
                self.set_register(to, value)?;
            }
            OP4_COND_EQ => {
                let register = op >> 4;
                let value = next_u32()?;
                let x = self.register(register)?;
                self.compare_state = u32::from(x == value);
            }
            OP4_COND_ANDEQ => {
                let register = op >> 4;
                let value = next_u32()?;
                let x = self.register(register)?;
                self.compare_state |= u32::from(x == value);
            }
            OP4_COND_JUMP => {
                if self.compare_state != 0 {
                    self.instruction_index = op >> 4;
                }
            }
            OP4_JUMP => {
                self.instruction_index = op >> 4;
            }
            OP4_CALL => {
                let address = op >> 4;
                *self.stack.get_mut(self.stack_index as usize).ok_or(Error)? =
                    self.instruction_index;
                self.stack_index += 1;
                self.instruction_index = address;
            }
            OP4_RET => {
                if self.stack_index == 0 {
                    return Ok(Yield::Finish);
                }
                self.stack_index -= 1;
                self.instruction_index = *self.stack.get(self.stack_index as usize).ok_or(Error)?;
            }
            OP4_SYS => {
                let id = op >> 4;
                return Ok(Yield::Sys { id });
            }
            OP4_ARRAY_SET => self.array_index = op >> 4,
            OP4_ARRAY_ADD => {
                let i = next_u32()?;
                self.array_index += (op >> 4) * self.register(i)?;
            }
            OP4_ARRAY_LOAD => {
                let x = self.register(self.array_index)?;
                self.set_register(op >> 4, x)?;
                self.array_index += 1;
            }
            OP4_ARRAY_STORE => {
                let x = self.register(op >> 4)?;
                self.set_register(self.array_index, x)?;
                self.array_index += 1;
            }
            _ => return Err(Error),
        }
        Ok(Yield::Preempt)
    }

    pub fn register(&self, offset: u32) -> Result<u32, Error> {
        self.registers.get(offset as usize).copied().ok_or(Error)
    }

    pub fn set_register(&mut self, offset: u32, value: u32) -> Result<(), Error> {
        self.registers
            .get_mut(offset as usize)
            .ok_or(Error)
            .map(|r| *r = value)
    }
}

impl OpEncoder {
    fn cur(&self) -> u32 {
        self.instructions.len().try_into().unwrap()
    }

    fn op_move(&mut self, from: u32, to: u32) {
        self.instructions
            .extend_from_slice(&[OP4_MOVE | (to << 4), from]);
    }

    fn op_set(&mut self, value: u32, to: u32) {
        self.instructions
            .extend_from_slice(&[OP4_SET | (to << 4), value]);
    }

    fn op_call(&mut self, address: u32) {
        self.resolve_label
            .try_insert(self.cur(), (Resolve::Shift4, address))
            .unwrap();
        self.instructions.push(OP4_CALL);
    }

    fn op_ret(&mut self) {
        self.instructions.push(OP4_RET);
    }

    fn op_jump(&mut self, address: u32) {
        self.resolve_label
            .try_insert(self.cur(), (Resolve::Shift4, address))
            .unwrap();
        self.instructions.push(OP4_JUMP);
    }

    fn op_cond_eq(&mut self, register: u32, value: u32) {
        self.instructions
            .extend_from_slice(&[OP4_COND_EQ | (register << 4), value]);
    }

    fn op_cond_andeq(&mut self, register: u32, value: u32) {
        self.instructions
            .extend_from_slice(&[OP4_COND_ANDEQ | (register << 4), value]);
    }

    fn op_cond_jump(&mut self, address: u32) {
        self.resolve_label
            .try_insert(self.cur(), (Resolve::Shift4, address))
            .unwrap();
        self.instructions.push(OP4_COND_JUMP);
    }

    fn op_sys(&mut self, id: u32) {
        assert!(id <= u32::MAX >> 4);
        self.instructions.extend_from_slice(&[OP4_SYS | (id << 4)]);
    }

    fn op_array_set(&mut self, base: u32) {
        assert!(base <= u32::MAX >> 4);
        self.instructions.push(OP4_ARRAY_SET | (base << 4))
    }

    fn op_array_add(&mut self, stride: u32, index: u32) {
        assert!(stride <= u32::MAX >> 4);
        self.instructions
            .extend_from_slice(&[OP4_ARRAY_ADD | (stride << 4), index]);
    }

    fn op_array_load(&mut self, register: u32) {
        assert!(register <= u32::MAX >> 4);
        self.instructions.push(OP4_ARRAY_LOAD | (register << 4))
    }

    fn op_array_store(&mut self, register: u32) {
        assert!(register <= u32::MAX >> 4);
        self.instructions.push(OP4_ARRAY_STORE | (register << 4))
    }

    fn finish(mut self, strings_buffer: &[u8]) -> (Box<[u32]>, u32) {
        for (&address, (resolve, label)) in self.resolve_label.iter() {
            let addr = self.label_to_address[label];
            match resolve {
                Resolve::Full => self.instructions[address as usize] = addr,
                Resolve::Shift4 => {
                    assert!(addr < u32::MAX >> 4);
                    self.instructions[address as usize] |= addr << 4;
                }
            }
        }
        let pc = self.cur();
        for (&i, ()) in self.resolve_constant.iter() {
            self.instructions[i as usize] += pc;
        }

        let strings_offset = self.cur();
        extend_u32_from_u8(&mut self.instructions, strings_buffer);

        (self.instructions.into(), strings_offset)
    }
}

fn words_for_bits(bits: u32) -> u32 {
    (bits + 31) / 32
}

fn words_for_bytes(bytes: u32) -> u32 {
    (bytes + 3) / 4
}

fn slice_u32_as_u8(s: &[u32]) -> &[u8] {
    let len = s.len() * 4;
    let ptr = s.as_ptr().cast::<u8>();
    unsafe { core::slice::from_raw_parts(ptr, len) }
}

fn slice_u32_as_u8_mut(s: &mut [u32]) -> &mut [u8] {
    let len = s.len() * 4;
    let ptr = s.as_mut_ptr().cast::<u8>();
    unsafe { core::slice::from_raw_parts_mut(ptr, len) }
}

fn extend_u32_from_u8(v: &mut Vec<u32>, s: &[u8]) {
    v.extend(s.chunks(4).map(|v| {
        let mut b = [0; 4];
        b[..v.len()].copy_from_slice(v);
        u32::from_le_bytes(b)
    }));
}
