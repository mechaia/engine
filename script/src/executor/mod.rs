use crate::Program;

pub mod wordvm;

#[derive(Debug)]
pub enum Executable {
    WordVM(wordvm::WordVM),
}

#[derive(Debug)]
pub enum Instance {
    WordVM(wordvm::WordVMState),
}

#[derive(Debug)]
pub enum Yield {
    Finish,
    Sys { id: u32 },
}

#[derive(Clone, Debug)]
pub struct Error;

impl Executable {
    pub fn from_program(program: &Program) -> Self {
        Self::WordVM(wordvm::WordVM::from_program(program))
    }

    pub fn new_instance(&self) -> Instance {
        match self {
            Self::WordVM(vm) => Instance::WordVM(vm.create_state()),
        }
    }

    pub fn const_str<'str>(&'str self, ptr: u32, len: u32) -> Result<&'str [u8], Error> {
        let ptr = usize::try_from(ptr).unwrap();
        let len = usize::try_from(len).unwrap();
        match self {
            Self::WordVM(vm) => vm
                .strings_buffer()
                .get(ptr..)
                .and_then(|s| s.get(..len))
                .ok_or(Error),
        }
    }
}

impl Instance {
    pub fn run(&mut self, executable: &Executable) -> Result<Yield, Error> {
        match (self, executable) {
            (Self::WordVM(state), Executable::WordVM(vm)) => loop {
                match state.step(vm)? {
                    wordvm::Yield::Finish => return Ok(Yield::Finish),
                    wordvm::Yield::Sys { id } => return Ok(Yield::Sys { id }),
                    wordvm::Yield::Preempt => {}
                }
            },
        }
    }

    pub fn inputs_slice(
        &self,
        executable: &Executable,
        sys: u32,
        buf: &mut [u32],
    ) -> Result<(), Error> {
        match (self, executable) {
            (Self::WordVM(state), Executable::WordVM(vm)) => {
                let (mut input, _) = vm.sys_registers(sys);
                if input == u32::MAX {
                    return Err(Error);
                }
                for e in buf.iter_mut() {
                    let reg = vm.constant(input)?;
                    *e = state.register(reg)?;
                    input += 1;
                }
                Ok(())
            }
        }
    }

    pub fn inputs<const N: usize>(
        &self,
        executable: &Executable,
        sys: u32,
    ) -> Result<[u32; N], Error> {
        let mut buf = [0; N];
        self.inputs_slice(executable, sys, &mut buf)?;
        Ok(buf)
    }

    pub fn outputs(
        &mut self,
        executable: &Executable,
        sys: u32,
        values: &[u32],
    ) -> Result<(), Error> {
        match (self, executable) {
            (Self::WordVM(state), Executable::WordVM(vm)) => {
                let (_, mut output) = vm.sys_registers(sys);
                if output == u32::MAX {
                    return Err(Error);
                }
                for e in values.iter() {
                    let reg = vm.constant(output)?;
                    state.set_register(reg, *e)?;
                    output += 1;
                }
                Ok(())
            }
        }
    }
}
