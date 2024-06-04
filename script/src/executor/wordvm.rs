use {
    crate::{
        program::{Constant, Function, Instruction, Register},
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
const OP4_JUMP: u32 = 2;
const OP4_JUMPEQ: u32 = 3;
const OP4_CALL: u32 = 4;
const OP4_RET: u32 = 5;
const OP4_SYS: u32 = 6;

const OP4_ARRAY_STORE: u32 = 7;
const OP4_ARRAY_LOAD: u32 = 8;

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
            registers_size += words_for_bits(reg.bits) * reg.length;
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

        let mut constants_size = 0;
        let mut constant_offsets = Vec::new();

        for cst in program.constants.iter() {
            constant_offsets.push(constants_size);
            constants_size += words_for_bytes(cst.value.len() as u32);
        }

        let mut conv = |i, f: &Function| {
            encoder
                .label_to_address
                .try_insert(i, encoder.cur())
                .unwrap();
            for instr in f.0.iter() {
                let r = |i: u32| register_offsets[i as usize];
                let ar = |i: u32| array_register_offsets[i as usize];
                let c = |i: u32| constant_offsets[i as usize];
                let l = |i: u32| words_for_bits(program.registers[i as usize].bits);
                match instr {
                    &Instruction::Move { to, from } => encoder.op_move(r(from), r(to), l(to)),
                    &Instruction::Set { to, from } => encoder.op_set(c(from), r(to), l(to)),
                    &Instruction::ToArray {
                        index,
                        array,
                        register,
                    } => encoder.op_array_store(r(index), ar(array), r(register), l(register)),
                    &Instruction::FromArray {
                        index,
                        array,
                        register,
                    } => encoder.op_array_load(r(index), ar(array), r(register), l(register)),
                    &Instruction::Call { address } => encoder.op_call(address),
                    &Instruction::Return => {
                        encoder.op_ret();
                        // break just in case another instruction follows
                        break;
                    }
                    &Instruction::Jump { address } => {
                        // If the target address is just behind this function,
                        // omit this redundant jump
                        if i + 1 != address || 1 == 0 {
                            encoder.op_jump(address);
                        }
                        // break just in case another instruction follows
                        break;
                    }
                    &Instruction::JumpEq {
                        address,
                        register,
                        constant,
                    } => {
                        // Constant value is inline, so don't map
                        encoder.op_jumpeq(address, r(register), constant)
                    }
                    &Instruction::SystemCall { id } => encoder.op_sys(id),
                }
            }
        };

        for (i, f) in program.functions.iter().enumerate() {
            let i = u32::try_from(i).unwrap();
            conv(i, f);
        }

        let (instructions, strings_offset) =
            encoder.finish(&program.constants, &program.strings_buffer);
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
        let mut cst_offset = self.strings_offset;

        let mut i = 0;
        loop {
            let Some(op) = self.instructions[..cst_offset as usize].get(i) else {
                break;
            };
            i += 1;
            match *op & 15 {
                OP4_SET => {
                    if let Some(&from) = self.instructions.get(i) {
                        cst_offset = cst_offset.min(from)
                    }
                    i += 2;
                }
                OP4_JUMP | OP4_CALL | OP4_RET | OP4_SYS => {}
                OP4_MOVE | OP4_JUMPEQ => i += 2,
                OP4_ARRAY_LOAD | OP4_ARRAY_STORE => i += 3,
                // break so the error is visible
                _ => {
                    cst_offset = self.strings_offset;
                    break;
                }
            }
        }

        write!(f, "stack size: {}\n", self.stack_size)?;
        write!(f, "registers size: {}\n", self.registers_size)?;
        write!(f, "constants offset: {cst_offset}\n")?;
        write!(f, "strings offset: {}\n", self.strings_offset)?;
        f.write_str("instructions:\n")?;
        let mut i = 0;
        loop {
            let Some(op) = self.instructions[..cst_offset as usize].get(i) else {
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
                    let count = op >> 4;
                    match (next_u32(), next_u32()) {
                        (Some(from), Some(to)) => write!(f, "MOVE    {count} {from} {to}")?,
                        _ => write!(f, "MOVE    {count} ???")?,
                    }
                }
                OP4_SET => {
                    let count = op >> 4;
                    match (next_u32(), next_u32()) {
                        (Some(from), Some(to)) => write!(f, "SET     {count} {from} {to}")?,
                        _ => write!(f, "SET     {count} ???")?,
                    }
                }
                OP4_SYS => {
                    let id = op >> 4;
                    write!(f, "SYS     {id}")?
                }
                OP4_JUMP => {
                    let address = op >> 4;
                    write!(f, "JUMP    {address}\n")?
                }
                OP4_JUMPEQ => {
                    let address = op >> 4;
                    match (next_u32(), next_u32()) {
                        (Some(register), Some(value)) => {
                            write!(f, "JUMPEQ  {address} {register} {value}")?;
                        }
                        _ => write!(f, "JUMPEQ  {address} ???")?,
                    }
                }
                OP4_CALL => {
                    let address = op >> 4;
                    write!(f, "CALL    {address}")?
                }
                OP4_RET => f.write_str("RET\n")?,
                o @ OP4_ARRAY_LOAD | o @ OP4_ARRAY_STORE => {
                    let postfix = ["LD", "ST"][usize::from(o == OP4_ARRAY_STORE)];
                    let count = op >> 4;
                    match (next_u32(), next_u32(), next_u32()) {
                        (Some(index), Some(array), Some(register)) => {
                            write!(f, "ARRAY{postfix} {count} {index} {array} {register}")?
                        }
                        _ => write!(f, "ARRAY{postfix} {count} ???")?,
                    }
                }
                op => write!(f, "??? {op}")?,
            }
            f.write_str("\n")?;
        }

        f.write_str("constants (right to left):\n")?;
        for c in self
            .instructions
            .iter()
            .take(self.strings_offset as usize)
            .skip(i)
        {
            let c = c.to_le();
            write!(f, "{i:>7}: {c:08x}\n")?;
            i += 1;
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
                let count = (op >> 4) as usize;
                let from = next_u32()? as usize;
                let to = next_u32()? as usize;
                if to > self.registers.len() || from > self.registers.len() {
                    return Err(Error);
                }
                let (from, to) = if from < to {
                    // from --- to
                    let (a, b) = self.registers.split_at_mut(to);
                    let a = a.get_mut(from..).ok_or(Error)?;
                    (a, b)
                } else {
                    // to -- from
                    let (a, b) = self.registers.split_at_mut(from);
                    let a = a.get_mut(to..).ok_or(Error)?;
                    (b, a)
                };
                let from = from.get(..count).ok_or(Error)?;
                let to = to.get_mut(..count).ok_or(Error)?;
                to.copy_from_slice(from);
            }
            OP4_SET => {
                let count = (op >> 4) as usize;
                let from = next_u32()? as usize;
                let to = next_u32()? as usize;

                let from = vm
                    .instructions
                    .get(from..)
                    .and_then(|s| s.get(..count))
                    .ok_or(Error)?;
                let to = self
                    .registers
                    .get_mut(to..)
                    .and_then(|s| s.get_mut(..count))
                    .ok_or(Error)?;

                to.copy_from_slice(from);
            }
            OP4_JUMP => {
                self.instruction_index = op >> 4;
            }
            OP4_JUMPEQ => {
                let address = op >> 4;
                let register = next_u32()? as usize;
                let value = next_u32()?;

                let cur = *self.registers.get(register as usize).ok_or(Error)?;
                if cur == value {
                    self.instruction_index = address;
                }
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
            OP4_ARRAY_LOAD => {
                let count = (op >> 4) as usize;
                let index = next_u32()? as usize;
                let base = next_u32()? as usize;
                let register = next_u32()? as usize;

                let from = base + (index * count);
                let to = register;

                self.registers.copy_within(from..from + count, to);
            }
            OP4_ARRAY_STORE => {
                let count = (op >> 4) as usize;
                let index = next_u32()? as usize;
                let base = next_u32()? as usize;
                let register = next_u32()? as usize;

                let to = base + (index * count);
                let from = register;

                self.registers.copy_within(from..from + count, to);
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

    fn op_move(&mut self, from: u32, to: u32, count: u32) {
        assert!(count <= u32::MAX >> 4);
        self.instructions
            .extend_from_slice(&[OP4_MOVE | (count << 4), from, to]);
    }

    fn op_set(&mut self, from: u32, to: u32, count: u32) {
        assert!(count <= u32::MAX >> 4);
        self.resolve_constant
            .try_insert(self.cur() + 1, ())
            .unwrap();
        self.instructions
            .extend_from_slice(&[OP4_SET | (count << 4), from, to]);
    }

    fn op_call(&mut self, address: u32) {
        self.resolve_label
            .try_insert(self.cur(), (Resolve::Shift4, address))
            .unwrap();
        self.instructions.extend_from_slice(&[OP4_CALL]);
    }

    fn op_ret(&mut self) {
        self.instructions.push(OP4_RET);
    }

    fn op_jump(&mut self, address: u32) {
        self.resolve_label
            .try_insert(self.cur(), (Resolve::Shift4, address))
            .unwrap();
        self.instructions.extend_from_slice(&[OP4_JUMP]);
    }

    fn op_jumpeq(&mut self, address: u32, register: u32, value: u32) {
        self.resolve_label
            .try_insert(self.cur(), (Resolve::Shift4, address))
            .unwrap();
        self.instructions
            .extend_from_slice(&[OP4_JUMPEQ, register, value]);
    }

    fn op_sys(&mut self, id: u32) {
        assert!(id <= u32::MAX >> 4);
        self.instructions.extend_from_slice(&[OP4_SYS | (id << 4)]);
    }

    fn op_array_load(&mut self, index: u32, base: u32, register: u32, count: u32) {
        assert!(count <= u32::MAX >> 4);
        self.instructions.extend_from_slice(&[
            OP4_ARRAY_LOAD | (count << 4),
            index,
            base,
            register,
        ]);
    }

    fn op_array_store(&mut self, index: u32, base: u32, register: u32, count: u32) {
        assert!(count <= u32::MAX >> 4);
        self.instructions.extend_from_slice(&[
            OP4_ARRAY_STORE | (count << 4),
            index,
            base,
            register,
        ]);
    }

    fn finish(mut self, constants: &[Constant], strings_buffer: &[u8]) -> (Box<[u32]>, u32) {
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

        for cst in constants.iter() {
            extend_u32_from_u8(&mut self.instructions, &cst.value);
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
