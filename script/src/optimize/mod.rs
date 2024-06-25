use {
    crate::{
        program::{debug, Function, Instruction, SwitchCase},
        Program,
    },
    core::mem,
    util::{bit::BitVec, LinearSet},
};

/// Only remove unused functions and registers.
pub fn remove_dead(program: &mut Program, mut debug: Option<&mut debug::Program>) {
    sort_pre_order(program, debug.as_deref_mut());
    remove_unused_sys(program);
    remove_unused_registers(program, debug.as_deref_mut());
}

/// Very basic and fast optimizer.
pub fn simple(program: &mut Program, mut debug: Option<&mut debug::Program>) {
    sort_pre_order(program, debug.as_deref_mut());
    inline(program, debug.as_deref_mut());
    sort_pre_order(program, debug.as_deref_mut());

    break_move_chains(program, debug.as_deref_mut());
    elide_redundant_moves(program, debug.as_deref_mut());

    remove_unused_sys(program);
    remove_unused_registers(program, debug.as_deref_mut());
}

/// Inline functions.
fn inline(program: &mut Program, mut debug: Option<&mut debug::Program>) {
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
            // Inline functions that are only moves or simple wrappers
            // These functions are usually simple constructors like vec3_new, which just copy registers
            let it = || b.instructions.iter();
            if it()
                .filter(|i| {
                    matches!(
                        i,
                        Instruction::RegisterLoad(_) | Instruction::RegisterStore(_)
                    )
                })
                .count()
                >= b.instructions.len() - 1
            {
                let max = usize::from(b.next == -1);
                if it().filter(|i| matches!(i, Instruction::Call(_))).count() <= max {
                    return true;
                }
            }
            false
        })
        .collect::<Vec<_>>();

    for i in (0..program.functions.len()).rev() {
        let mut new_instrs = Vec::new();
        let mut new_lines = Vec::new();

        let debug_f = debug.as_deref().map(|d| &d.functions[i]);

        match &program.functions[i] {
            Function::Block(b) => {
                debug
                    .as_deref()
                    .map(|d| &d.functions[i])
                    .map(|d| assert_eq!(b.instructions.len(), d.instruction_to_line.len()));
                for (instr_i, &instr) in b.instructions.iter().enumerate() {
                    let mut i2l = debug_f.map(|d| d.instruction_to_line[instr_i]);
                    match instr {
                        Instruction::Call(mut address) => {
                            // inline recursively to avoid inserting redundant calls
                            loop {
                                debug_assert_ne!(address as usize, i);

                                if address < 0 || !should_inline[address as usize] {
                                    if address != -1 {
                                        new_instrs.push(Instruction::Call(address));
                                        i2l.map(|l| new_lines.push(l));
                                    }
                                    break;
                                }

                                let f_2 = &program.functions[address as usize];
                                let Function::Block(b_2) = f_2 else { todo!() };

                                new_instrs.extend(b_2.instructions.iter());
                                if let Some(d) =
                                    debug.as_deref().map(|d| &d.functions[address as usize])
                                {
                                    new_lines.extend(d.instruction_to_line.iter());
                                    i2l = Some(d.last_line);
                                }
                                assert_eq!(new_instrs.len(), new_lines.len());

                                address = b_2.next;
                            }
                        }
                        instr => {
                            new_instrs.push(instr);
                            i2l.map(|l| new_lines.push(l));
                        }
                    }
                }

                let Function::Block(b) = &mut program.functions[i] else {
                    unreachable!()
                };
                let mut d = debug.as_deref_mut().map(|d| &mut d.functions[i]);

                while b.next == -1 {
                    match new_instrs.pop() {
                        Some(Instruction::Call(address)) => {
                            b.next = address;
                            d.as_deref_mut()
                                .map(|d| d.last_line = new_lines.pop().unwrap());
                        }
                        Some(n) => {
                            new_instrs.push(n);
                            break;
                        }
                        None => break,
                    }
                }

                b.instructions = new_instrs.into();
                d.as_deref_mut()
                    .map(|d| d.instruction_to_line = new_lines.into());

                d.map(|d| assert_eq!(b.instructions.len(), d.instruction_to_line.len()));
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
}

/// Determine per-function reference counts.
fn count_function_references(program: &Program) -> Vec<u32> {
    let mut func_refcounts = vec![0; program.functions.len()];
    for f in program.functions.iter() {
        visit_funcref(f, |x| {
            if x >= 0 {
                func_refcounts[x as usize] += 1
            }
        });
    }
    func_refcounts[0] = u32::MAX;
    func_refcounts
}

/// Sort functions in pre-order, starting from the first function.
///
/// This also removes unreferenced functions.
fn sort_pre_order(program: &mut Program, debug: Option<&mut debug::Program>) {
    fn f(new_list: &mut Vec<i32>, program: &Program, index: i32, visited: &mut BitVec) {
        if index < 0 || visited.get(index as usize).unwrap() {
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

    let mut remap_table = vec![i32::MAX; program.functions.len()];

    for (i, k) in new_list.iter().enumerate() {
        remap_table[*k as usize] = i as i32;
    }

    program.functions = new_list
        .iter()
        .map(|&i| {
            let mut f = mem::take(&mut program.functions[i as usize]);
            visit_funcref_mut(&mut f, |x| {
                if *x >= 0 {
                    *x = remap_table[*x as usize]
                }
            });
            f
        })
        .collect();

    if let Some(debug) = debug {
        debug.functions = new_list
            .iter()
            .map(|&i| mem::take(&mut debug.functions[i as usize]))
            .collect();
    }
}

/// Visit all function references of the given function, i.e. calls and jumps.
fn visit_funcref<F: FnMut(i32)>(function: &Function, mut f: F) {
    match function {
        Function::Block(b) => {
            f(b.next);
            for instr in b.instructions.iter() {
                if let Instruction::Call(address) = instr {
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
fn visit_funcref_mut<F: FnMut(&mut i32)>(function: &mut Function, mut f: F) {
    match function {
        Function::Block(b) => {
            f(&mut b.next);
            for instr in b.instructions.iter_mut() {
                if let Instruction::Call(address) = instr {
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
        let mut test_set = |a: i32| {
            if a < 0 {
                referenced.set(!a as usize, true);
            }
        };
        for instr in b.instructions.iter() {
            let Instruction::Call(address) = instr else {
                continue;
            };
            test_set(*address);
        }
        test_set(b.next);
    }
    for (b, s) in referenced.iter().zip(program.sys_to_registers.iter_mut()) {
        if !b {
            *s = None;
        }
    }
}

/// Remove unused registers
fn remove_unused_registers(program: &mut Program, debug: Option<&mut debug::Program>) {
    let mut referenced = BitVec::filled(program.registers.len(), false);
    let mut array_referenced = BitVec::filled(program.array_registers.len(), false);

    for f in program.functions.iter() {
        match f {
            Function::Block(b) => {
                for instr in b.instructions.iter() {
                    match instr {
                        Instruction::RegisterLoad(r)
                        | Instruction::RegisterStore(r)
                        | Instruction::ArrayStore(r)
                        | Instruction::ArrayIndex(r) => referenced.set(*r as usize, true),
                        &Instruction::ArrayAccess(array) => {
                            array_referenced.set(array as usize, true)
                        }
                        &Instruction::Call(address) => {
                            if address < 0 {
                                program.sys_to_registers[!address as usize]
                                    .iter()
                                    .flat_map(|m| m.iter_all())
                                    .for_each(|r| referenced.set(r as usize, true));
                            }
                        }
                        Instruction::ConstantLoad(_) => {}
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
                        Instruction::ArrayAccess(array) => {
                            *array = array_remap[*array as usize];
                        }
                        Instruction::RegisterStore(reg)
                        | Instruction::RegisterLoad(reg)
                        | Instruction::ArrayStore(reg)
                        | Instruction::ArrayIndex(reg) => {
                            *reg = remap[*reg as usize];
                        }
                        Instruction::ConstantLoad(_) | Instruction::Call(_) => {}
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

    if let Some(d) = debug {
        d.registers = referenced
            .iter()
            .enumerate()
            .filter(|x| x.1)
            .map(|x| mem::take(&mut d.registers[x.0]))
            .collect();
        d.array_registers = array_referenced
            .iter()
            .enumerate()
            .filter(|x| x.1)
            .map(|x| mem::take(&mut d.array_registers[x.0]))
            .collect();
    }
}

/// Flatten chain of jumps
/// e.g. `JUMP X -> JUMP Y -> JUMP Z` becomes `JUMP Z`
fn flatten_jump(program: &Program, mut start: i32) -> i32 {
    while start >= 0 {
        let Function::Block(b) = &program.functions[start as usize] else {
            break;
        };
        if !b.instructions.is_empty() {
            break;
        }
        start = b.next;
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

/// Break chains of moves.
///
/// This allows removing redundant moves and sets in a later pass.
fn break_move_chains(program: &mut Program, _debug: Option<&mut debug::Program>) {
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
        remove_aliases(values, reverse_alias, register);
    }

    fn remove_aliases(
        values: &mut [RegisterValue],
        reverse_alias: &mut [LinearSet<u32>],
        register: u32,
    ) {
        for reg in reverse_alias[register as usize].drain() {
            values[reg as usize] = RegisterValue::Unknown;
        }
    }

    for f in program.functions.iter_mut() {
        let Function::Block(b) = f else { continue };
        clear(&mut values, &mut reverse_alias);

        let mut cur_value = RegisterValue::Unknown;

        for instr in b.instructions.iter_mut() {
            match instr {
                Instruction::RegisterStore(to) => {
                    // remove aliases to this register
                    // do NOT try to be clever and "short-circuit" aliases in values[...]!!
                    // it will likely break with e.g. swizzles
                    // we may also miss some optimizations if we try
                    // instead, keep values[...] a chain and resolve iteratively
                    set_unknown(&mut values, &mut reverse_alias, *to);
                    values[*to as usize] = cur_value;
                    if let RegisterValue::Alias(from) = cur_value {
                        reverse_alias[from as usize].insert(*to);
                    }
                }
                Instruction::ConstantLoad(cst) => cur_value = RegisterValue::Constant(*cst),
                Instruction::RegisterLoad(from) => {
                    cur_value = RegisterValue::Alias(*from);
                    while let RegisterValue::Alias(v) = cur_value {
                        match values[v as usize] {
                            RegisterValue::Unknown => break,
                            v => cur_value = v,
                        }
                    }
                }
                Instruction::ArrayStore(r) | Instruction::ArrayIndex(r) => {
                    match values[*r as usize] {
                        RegisterValue::Unknown | RegisterValue::Constant(_) => {}
                        RegisterValue::Alias(reg) => {
                            *r = reg;
                        }
                    }
                }
                Instruction::Call(address) => {
                    if *address < 0 {
                        let sys = program.sys_to_registers[!*address as usize]
                            .as_ref()
                            .unwrap();
                        for &i in sys.outputs.iter() {
                            set_unknown(&mut values, &mut reverse_alias, i)
                        }
                    } else {
                        clear(&mut values, &mut reverse_alias);
                    }
                }
                Instruction::ArrayAccess(_) => cur_value = RegisterValue::Unknown,
            }
        }
    }
}

/// Remove moves and sets with no observable effect.
fn elide_redundant_moves(program: &mut Program, mut debug: Option<&mut debug::Program>) {
    let mut set = BitVec::filled(program.registers.len(), false);
    let used = find_read_registers(program);

    for (i_f, f) in program.functions.iter_mut().enumerate() {
        let Function::Block(b) = f else { continue };
        set.fill(false);
        let mut new_instrs = Vec::new();
        let mut new_lines = Vec::new();

        let mut debug_f = debug.as_deref_mut().map(|d| &mut d.functions[i_f]);

        let mut keep_accessor = true;
        for (i_instr, &instr) in b.instructions.iter().enumerate().rev() {
            let mut test_set = |r| used.get(r as usize).unwrap() && !set.replace(r as usize, true);
            match instr {
                Instruction::ConstantLoad(_) | Instruction::ArrayAccess(_) => {
                    if keep_accessor {
                        new_instrs.push(instr);
                    }
                }
                Instruction::ArrayIndex(index) => {
                    if keep_accessor {
                        new_instrs.push(instr);
                    }
                    set.set(index as usize, false);
                }
                Instruction::RegisterStore(reg) => {
                    keep_accessor = test_set(reg);
                    if keep_accessor {
                        new_instrs.push(instr);
                    }
                }
                Instruction::RegisterLoad(reg) => {
                    if keep_accessor {
                        if !matches!(new_instrs.last(), Some(Instruction::RegisterStore(r)) if *r == reg)
                        {
                            new_instrs.push(instr);
                        }
                    }
                    set.set(reg as usize, false);
                }
                Instruction::Call(address) => {
                    if address < 0 {
                        let map = program.sys_to_registers[!address as usize]
                            .as_ref()
                            .unwrap();
                        map.outputs.iter().for_each(|&r| set.set(r as usize, true));
                        map.inputs.iter().for_each(|&r| set.set(r as usize, false));
                    } else {
                        set.fill(false);
                    }
                    new_instrs.push(instr);
                }
                // TODO
                Instruction::ArrayStore(register) => {
                    set.set(register as usize, false);
                    new_instrs.push(instr);
                    keep_accessor = true;
                }
            }
            debug_f
                .as_deref_mut()
                .map(|d| new_lines.resize(new_instrs.len(), d.instruction_to_line[i_instr]));
        }
        new_instrs.reverse();
        new_lines.reverse();
        b.instructions = new_instrs.into();
        debug_f.map(|d| d.instruction_to_line = new_lines.into());
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
                        Instruction::RegisterLoad(r)
                        | Instruction::ArrayStore(r)
                        | Instruction::ArrayIndex(r) => set.set(r as usize, true),
                        Instruction::Call(id) => {
                            if id < 0 {
                                program.sys_to_registers[!id as usize]
                                    .as_ref()
                                    .unwrap()
                                    .inputs
                                    .iter()
                                    .for_each(|&r| set.set(r as usize, true));
                            }
                        }
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
