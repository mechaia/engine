pub mod wordvm;

/*
pub struct Executor<'a> {
    pc: u32,
    registers: Box<[u32]>,
    call_stack: Vec<u32>,
    program: &'a Program,
}

#[derive(Debug)]
pub enum Yield {
    Finish,
    Call { function: u32 },
    Limit,
}

impl<'a> Executor<'a> {
    pub fn new(program: &'a Program, entry: Str) -> Result<Self, Error> {
        let entry = program.entry_points.get(&entry).ok_or(Error)?;
        todo!()
    }

    pub fn run(&mut self, max_steps: Option<NonZeroU64>) -> Result<Yield, Error> {
        for _ in 0..max_steps.unwrap_or(NonZeroU64::MAX).get() {
            match self.next_u8()? {
                OP_CALL => {
                    let pc = self.next_i32()?;
                    let next_pc = self.pc + 4;
                    if let Ok(pc) = u32::try_from(pc) {
                        self.pc = pc;
                        *self.call_stack.get_mut(self.call_stack_index as usize).ok_or(Error)? = next_pc;
                        self.call_stack_index += 1;
                    } else {
                        self.pc = next_pc;
                        return Ok(Yield::Call { function: (!pc).try_into().unwrap() });
                    }
                },
                OP_CALLTBL => {
                    let count = self.next_u32()?;
                    let next_pc = self.pc + count * 4;
                    let pc = self.get_i32(self.pc + self.acc * 4)?;
                    if let Ok(pc) = u32::try_from(pc) {
                        self.pc = pc;
                        *self.call_stack.get_mut(self.call_stack_index as usize).ok_or(Error)? = next_pc;
                        self.call_stack_index += 1;
                    } else {
                        self.pc = next_pc;
                        return Ok(Yield::Call { function: (!pc).try_into().unwrap() });
                    }
                }
                OP_RET => {
                    self.call_stack_index -= 1;
                    self.pc = *self.call_stack.get(self.call_stack_index as usize).ok_or(Error)?;
                }
                OP_SET => {
                    self.acc = self.next_u32()?;
                }
                OP_STORE => {
                    let addr = self.next_u32()? as usize;
                    let v = self.data.get_mut(addr..addr + 4).ok_or(Error)?;
                    v.copy_from_slice(&self.acc.to_ne_bytes())
                }
                OP_LOAD => {
                    let addr = self.next_u32()? as usize;
                    let v = self.data.get(addr..addr + 4).ok_or(Error)?;
                    self.acc = u32::from_ne_bytes(v.try_into().unwrap());
                }
                OP_ADD => {
                    self.acc += self.next_u32()?;
                }
                _ => return Err(Error),
            }
        }
        Ok(Yield::Limit)
    }

    fn next_u8(&mut self) -> Result<u8, Error> {
        let pc = self.pc as usize;
        self.pc += 1;
        self.program.instructions.get(pc).ok_or(Error).copied()
    }

    fn next_u32(&mut self) -> Result<u32, Error> {
        let pc = self.pc;
        self.pc += 4;
        self.get_u32(pc)
    }

    fn next_i32(&mut self) -> Result<i32, Error> {
        self.next_u32().map(|v| v as _)
    }

    fn get_u32(&mut self, index: u32) -> Result<u32, Error> {
        self.program
            .instructions
            .get(index as _..index as usize + 4)
            .map(|s| u32::from_ne_bytes(s.try_into().unwrap()))
            .ok_or(Error)
    }

    fn get_i32(&mut self, index: u32) -> Result<i32, Error> {
        self.get_u32(index as _).map(|v| v as _)
    }
}
*/
