use {
    crate::{
        program::{Constant, Function, Instruction, SwitchCase},
        Map, Program,
    },
    core::mem,
    util::bit::BitVec,
};

/// Very basic and fast optimizer.
pub fn simple(program: &mut Program) {
    sort_pre_order(program);

    let func_refcounts = count_function_references(program);

    // inline small functions and functions only referenced once by a direct call/jump
    // do dead code elimination while at it
    let should_inline = program
        .functions
        .iter()
        .zip(&func_refcounts)
        .map(|(f, rc)| {
            let Function::Block(b) = f else { return false };
            if *rc <= 1 {
                return true;
            }
            if b.instructions.is_empty() {
                return true;
            }
            if b.next.is_none() {
                return matches!(
                    &*b.instructions,
                    [Instruction::Sys { .. }] | [Instruction::Call { .. }]
                );
            }
            false
        })
        .collect::<Vec<_>>();

    for i in (0..program.functions.len()).rev() {
        let mut new_instrs = Vec::new();

        match &program.functions[i] {
            Function::Block(b) => {
                for &instr in b.instructions.iter() {
                    match instr {
                        Instruction::Call { mut address } => {
                            // inline recursively to avoid inserting redundant calls
                            loop {
                                debug_assert_ne!(address as usize, i);

                                if !should_inline[address as usize] {
                                    new_instrs.push(Instruction::Call { address });
                                    break;
                                }

                                let f_2 = &program.functions[address as usize];
                                let Function::Block(b_2) = f_2 else { todo!() };

                                new_instrs.extend_from_slice(&b_2.instructions);

                                if let Some(next_2) = b_2.next {
                                    address = next_2
                                } else {
                                    break;
                                }
                            }
                        }
                        instr => new_instrs.push(instr),
                    }
                }

                let Function::Block(b) = &mut program.functions[i] else {
                    unreachable!()
                };

                if b.next.is_none() {
                    match new_instrs.pop() {
                        Some(Instruction::Call { address }) => b.next = Some(address),
                        Some(n) => new_instrs.push(n),
                        None => {}
                    }
                }

                b.instructions = new_instrs.into();
            }
            Function::Switch(s) => {
                let cases = s
                    .cases
                    .iter()
                    .map(|c| SwitchCase {
                        constant: c.constant,
                        function: flatten_jump(program, c.function),
                    })
                    .collect();
                let default = s.default.map(|a| flatten_jump(program, a));
                let Function::Switch(s) = &mut program.functions[i] else {
                    unreachable!()
                };
                s.cases = cases;
                s.default = default;
            }
        }
    }

    sort_pre_order(program);

    break_move_chains(program);
    elide_redundant_moves(program);

    remove_unused_sys(program);
    remove_unused_registers(program);
}

/// Determine per-function reference counts.
fn count_function_references(program: &Program) -> Vec<u32> {
    let mut func_refcounts = vec![0; program.functions.len()];
    for f in program.functions.iter() {
        visit_funcref(f, |x| func_refcounts[x as usize] += 1);
    }
    func_refcounts[0] = u32::MAX;
    func_refcounts
}

/// Sort functions in pre-order, starting from the first function.
///
/// This also removes unreferenced functions.
fn sort_pre_order(program: &mut Program) {
    fn f(new_list: &mut Vec<u32>, program: &Program, index: u32, visited: &mut BitVec) {
        if visited.get(index as usize).unwrap() {
            return;
        }
        visited.set(index as usize, true);
        new_list.push(index);
        visit_funcref(&program.functions[index as usize], |x| {
            f(new_list, program, x, visited)
        });
    }

    let mut visited = BitVec::filled(program.functions.len(), false);
    let mut new_list = Vec::new();
    f(&mut new_list, program, 0, &mut visited);

    let mut remap_table = vec![u32::MAX; program.functions.len()];

    for (i, k) in new_list.iter().enumerate() {
        remap_table[*k as usize] = i as u32;
    }

    program.functions = new_list
        .iter()
        .map(|&index| {
            let mut f = mem::take(&mut program.functions[index as usize]);
            visit_funcref_mut(&mut f, |x| *x = remap_table[*x as usize]);
            f
        })
        .collect();
}

/// Visit all function references of the given function, i.e. calls and jumps.
fn visit_funcref<F: FnMut(u32)>(function: &Function, mut f: F) {
    match function {
        Function::Block(b) => {
            if let Some(x) = b.next {
                f(x)
            }
            for instr in b.instructions.iter() {
                if let Instruction::Call { address } = instr {
                    f(*address)
                }
            }
        }
        Function::Switch(s) => {
            if let Some(x) = s.default {
                f(x)
            }
            for c in s.cases.iter() {
                f(c.function)
            }
        }
    }
}

/// Visit all function references of the given function, i.e. calls and jumps.
fn visit_funcref_mut<F: FnMut(&mut u32)>(function: &mut Function, mut f: F) {
    match function {
        Function::Block(b) => {
            if let Some(x) = b.next.as_mut() {
                f(x)
            }
            for instr in b.instructions.iter_mut() {
                if let Instruction::Call { address } = instr {
                    f(address)
                }
            }
        }
        Function::Switch(s) => {
            if let Some(x) = s.default.as_mut() {
                f(x)
            }
            for c in s.cases.iter_mut() {
                f(&mut c.function)
            }
        }
    }
}

/// Remove unused system calls
fn remove_unused_sys(program: &mut Program) {
    let mut referenced = BitVec::filled(program.sys_to_registers.len(), false);
    for f in program.functions.iter() {
        let Function::Block(b) = f else { continue };
        for instr in b.instructions.iter() {
            let Instruction::Sys { id } = instr else {
                continue;
            };
            referenced.set(usize::try_from(*id).unwrap(), true);
        }
    }
    for (b, s) in referenced.iter().zip(program.sys_to_registers.iter_mut()) {
        if !b {
            *s = None;
        }
    }
}

/// Remove unused registers
fn remove_unused_registers(program: &mut Program) {
    let mut referenced = BitVec::filled(program.registers.len(), false);
    let mut array_referenced = BitVec::filled(program.array_registers.len(), false);

    for f in program.functions.iter() {
        match f {
            Function::Block(b) => {
                for instr in b.instructions.iter() {
                    match instr {
                        &Instruction::Move { to, from } => {
                            referenced.set(to as usize, true);
                            referenced.set(from as usize, true);
                        }
                        &Instruction::ArrayAccess { array } => {
                            array_referenced.set(array as usize, true)
                        }
                        Instruction::ArrayStore { register: reg }
                        | Instruction::ArrayLoad { register: reg }
                        | Instruction::Set { to: reg, .. }
                        | Instruction::ArrayIndex { index: reg } => {
                            referenced.set(*reg as usize, true)
                        }
                        &Instruction::Sys { id } => {
                            program.sys_to_registers[id as usize]
                                .iter()
                                .flat_map(|m| m.iter_all())
                                .for_each(|r| referenced.set(r as usize, true));
                        }
                        Instruction::Call { .. } => {}
                    }
                }
            }
            Function::Switch(s) => {
                referenced.set(s.register as usize, true);
            }
        }
    }

    let regs = mem::take(&mut program.registers).into_vec();
    let mut new_regs = Vec::new();
    for (b, reg) in referenced.iter().zip(regs) {
        if b {
            new_regs.push(reg);
        }
    }
    program.registers = new_regs.into();

    let regs = mem::take(&mut program.array_registers).into_vec();
    let mut new_regs = Vec::new();
    for (b, reg) in array_referenced.iter().zip(regs) {
        if b {
            new_regs.push(reg);
        }
    }
    program.array_registers = new_regs.into();

    let mut remap = Vec::with_capacity(referenced.len());
    let mut array_remap = Vec::with_capacity(array_referenced.len());

    let mut i = 0;
    for b in referenced.iter() {
        remap.push(if b { i } else { u32::MAX });
        i += u32::from(b);
    }

    let mut i = 0;
    for b in array_referenced.iter() {
        array_remap.push(if b { i } else { u32::MAX });
        i += u32::from(b);
    }

    for f in program.functions.iter_mut() {
        match f {
            Function::Block(b) => {
                for instr in b.instructions.iter_mut() {
                    match instr {
                        Instruction::Move { to, from } => {
                            *to = remap[*to as usize];
                            *from = remap[*from as usize];
                        }
                        Instruction::ArrayAccess { array } => {
                            *array = array_remap[*array as usize];
                        }
                        Instruction::ArrayStore { register: reg }
                        | Instruction::ArrayLoad { register: reg }
                        | Instruction::Set { to: reg, .. }
                        | Instruction::ArrayIndex { index: reg } => {
                            *reg = remap[*reg as usize];
                        }
                        Instruction::Sys { .. } | Instruction::Call { .. } => {}
                    }
                }
            }
            Function::Switch(s) => {
                s.register = remap[s.register as usize];
            }
        }
    }

    for sys in program.sys_to_registers.iter_mut() {
        let Some(m) = sys else { continue };
        let mut it = m.iter_all_mut();
        while let Some(reg) = it.next() {
            *reg = remap[*reg as usize];
            assert_ne!(*reg, u32::MAX);
        }
    }
}

/// Flatten chain of jumps
/// e.g. `JUMP X -> JUMP Y -> JUMP Z` becomes `JUMP Z`
fn flatten_jump(program: &Program, mut start: u32) -> u32 {
    loop {
        let Function::Block(b) = &program.functions[start as usize] else {
            break;
        };
        if !b.instructions.is_empty() {
            break;
        }
        let Some(addr) = b.next else { break };
        start = addr;
    }
    start
}

#[derive(Clone, Copy)]
enum RegisterValue {
    /// Value is unknown, could be anything.
    Unknown,
    /// Value is a constant.
    Constant(u32),
    /// Value mirrors a register.
    Alias(u32),
}

#[derive(Clone, Debug, Default)]
struct LinearSet<T> {
    set: Vec<T>,
}

impl<T: Eq> LinearSet<T> {
    fn insert(&mut self, value: T) {
        if !self.set.contains(&value) {
            self.set.push(value);
        }
    }

    fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.set.drain(..)
    }

    fn clear(&mut self) {
        self.set.clear();
    }
}

/// Break chains of moves.
///
/// This allows removing redundant moves and sets in a later pass.
fn break_move_chains(program: &mut Program) {
    let it = || 0..program.registers.len();
    let mut values = it().map(|_| RegisterValue::Unknown).collect::<Box<_>>();
    let mut reverse_alias = it().map(|_| LinearSet::default()).collect::<Box<[_]>>();

    fn clear(values: &mut [RegisterValue], reverse_alias: &mut [LinearSet<u32>]) {
        values.iter_mut().for_each(|v| *v = RegisterValue::Unknown);
        reverse_alias.iter_mut().for_each(|v| v.clear());
    }

    fn set_alias(
        values: &mut [RegisterValue],
        reverse_alias: &mut [LinearSet<u32>],
        from: u32,
        to: u32,
    ) {
        values[to as usize] = RegisterValue::Alias(from);
        reverse_alias[from as usize].insert(to);
    }

    fn set_unknown(
        values: &mut [RegisterValue],
        reverse_alias: &mut [LinearSet<u32>],
        register: u32,
    ) {
        values[register as usize] = RegisterValue::Unknown;
        for reg in reverse_alias[register as usize].drain() {
            values[reg as usize] = RegisterValue::Unknown;
        }
    }

    for f in program.functions.iter_mut() {
        let Function::Block(b) = f else { continue };
        clear(&mut values, &mut reverse_alias);
        for instr in b.instructions.iter_mut() {
            match instr {
                &mut Instruction::Set { to, from } => {
                    values[to as usize] = RegisterValue::Constant(from);
                }
                &mut Instruction::Move { to, from } => match values[from as usize] {
                    RegisterValue::Unknown => {
                        set_alias(&mut values, &mut reverse_alias, from, to);
                    }
                    RegisterValue::Constant(from) => {
                        values[to as usize] = RegisterValue::Constant(from);
                        *instr = Instruction::Set { to, from }
                    }
                    RegisterValue::Alias(from) => {
                        set_alias(&mut values, &mut reverse_alias, from, to);
                        *instr = Instruction::Move { to, from }
                    }
                },
                Instruction::ArrayStore { register: r } | Instruction::ArrayIndex { index: r } => {
                    match values[*r as usize] {
                        RegisterValue::Unknown | RegisterValue::Constant(_) => {}
                        RegisterValue::Alias(reg) => *r = reg,
                    }
                }
                Instruction::Call { .. } => {
                    clear(&mut values, &mut reverse_alias);
                }
                &mut Instruction::Sys { id } => {
                    let sys = program.sys_to_registers[id as usize].as_ref().unwrap();
                    for &i in sys.outputs.iter() {
                        set_unknown(&mut values, &mut reverse_alias, i)
                    }
                }
                Instruction::ArrayAccess { .. } => {}
                Instruction::ArrayLoad { register } => {
                    set_unknown(&mut values, &mut reverse_alias, *register)
                }
            }
        }
    }
}

/// Remove moves and sets with no observable effect.
fn elide_redundant_moves(program: &mut Program) {
    let mut set = BitVec::filled(program.registers.len(), false);
    let used = find_read_registers(program);

    for f in program.functions.iter_mut() {
        let Function::Block(b) = f else { continue };
        set.fill(false);
        let mut new_instrs = Vec::new();
        let mut keep_array_access = true;
        for &instr in b.instructions.iter().rev() {
            let mut test_set = |r| used.get(r as usize).unwrap() && !set.replace(r as usize, true);
            match instr {
                Instruction::Move { to, from } if to == from => {}
                Instruction::Set { to, .. } | Instruction::Move { to, .. } => {
                    if test_set(to) {
                        new_instrs.push(instr);
                    }
                }
                Instruction::ArrayLoad { register } => {
                    keep_array_access = test_set(register);
                    if keep_array_access {
                        new_instrs.push(instr);
                    }
                }
                Instruction::ArrayIndex { .. } | Instruction::ArrayAccess { .. } => {
                    if keep_array_access {
                        new_instrs.push(instr);
                    }
                }
                Instruction::Call { .. } => {
                    set.fill(false);
                    new_instrs.push(instr);
                }
                Instruction::Sys { id } => {
                    let map = program.sys_to_registers[id as usize].as_ref().unwrap();
                    map.outputs.iter().for_each(|&r| set.set(r as usize, true));
                    map.inputs.iter().for_each(|&r| set.set(r as usize, false));
                    new_instrs.push(instr);
                }
                _ => new_instrs.push(instr),
            }
        }
        new_instrs.reverse();
        b.instructions = new_instrs.into();
    }
}

/// Mark registers which are read from.
///
/// "write-only" registers are unmarked.
fn find_read_registers(program: &Program) -> BitVec {
    let mut set = BitVec::filled(program.registers.len(), false);

    for f in program.functions.iter() {
        match f {
            Function::Block(b) => {
                for &instr in b.instructions.iter() {
                    match instr {
                        Instruction::Move { from: r, .. }
                        | Instruction::ArrayStore { register: r }
                        | Instruction::ArrayIndex { index: r } => set.set(r as usize, true),
                        Instruction::Sys { id } => program.sys_to_registers[id as usize]
                            .as_ref()
                            .unwrap()
                            .inputs
                            .iter()
                            .for_each(|&r| set.set(r as usize, true)),
                        _ => {}
                    }
                }
            }
            Function::Switch(s) => {
                set.set(s.register as usize, true);
            }
        }
    }

    set
}
