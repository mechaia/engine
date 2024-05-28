use {crate::{program::{Constant, Instruction}, Map, Program}, core::mem};

/// Very basic and fast optimizer.
pub fn simple(program: &mut Program) {
    let func_refcounts = count_function_references(program);

    // inline small functions and functions only referenced once by a direct call/jump
    // do dead code elimination while at it
    let should_inline = program.functions.iter().zip(&func_refcounts).map(|(f, rc)| {
        if f.0.iter().any(|i| matches!(i, Instruction::JumpEq { .. })) {
            return false;
        }
        if *rc <= 1 {
            // FIXME loops
            return true;
        }
        matches!(&*f.0, [Instruction::SystemCall { .. }, Instruction::Return])
    }).collect::<Vec<_>>();
    //let should_inline = program.functions.iter().zip(&func_refcounts).map(|(f, rc)| *rc <= 1).collect::<Vec<_>>();
    for i in 0..program.functions.len() {
        let mut new_instrs = Vec::new();
        'l1: for &instr in program.functions[i].0.iter() {
            dbg!(instr);
            match instr {
                Instruction::Call { address } | Instruction::Jump { address } if should_inline[address as usize] => {
                    let is_call = matches!(instr, Instruction::Call { .. });
                    'l2: for &instr_2 in program.functions[address as usize].0.iter() {
                        dbg!(instr_2);
                        match instr_2 {
                            Instruction::Jump { address } => {
                                if is_call {
                                    // replace with call so we return as original
                                    new_instrs.push(Instruction::Call { address });
                                    break 'l2;
                                } else {
                                    // keep tail call and break since all code behind this is dead
                                    new_instrs.push(Instruction::Jump { address });
                                    break 'l1;
                                }
                            }
                            // dead end of callee
                            Instruction::Return => {
                                if is_call {
                                    // not a dead end of caller, continue
                                    break 'l2
                                } else {
                                    // dead end of caller, return
                                    new_instrs.push(Instruction::Return);
                                    break 'l1
                                }
                            }
                            instr_2 => new_instrs.push(instr_2),
                        }
                    }
                }
                instr => new_instrs.push(instr)
            }
        }
        dbg!(&program.functions[i].0);
        dbg!(&new_instrs);
        program.functions[i].0 = new_instrs.into()
    }

    remove_dead(program);
}

/// Remove dead functions
fn remove_dead(program: &mut Program) {
    let func_refcounts = count_function_references(program);
    let mut remap_table = vec![u32::MAX; program.functions.len()];
    let mut remap_index = 0;
    let mut new_functions = Vec::new();
    for ((f, &rc), remap) in mem::take(&mut program.functions).into_vec().into_iter().zip(&func_refcounts).zip(&mut remap_table) {
        if rc == 0 {
            continue;
        }
        *remap = remap_index;
        remap_index += 1;
        new_functions.push(f);
    }
    dbg!(&remap_table);
    for f in new_functions.iter_mut() {
        for instr in f.0.iter_mut() {
            match instr {
                Instruction::Call { address } | Instruction::Jump { address } | Instruction::JumpEq { address, .. } => {
                    *address = remap_table[*address as usize]
                }
                _ => {}
            }
        }
    }
    program.functions = new_functions.into();
}

/// Determine per-function reference counts.
fn count_function_references(program: &Program) -> Vec<u32> {
    let mut func_refcounts = vec![0; program.functions.len()];
    for f in program.functions.iter() {
        for i in f.0.iter() {
            match i {
                &Instruction::Call { address } | &Instruction::Jump { address } | &Instruction::JumpEq { address, .. } => {
                    func_refcounts[address as usize] += 1;
                }
                _ => {}
            }
        }
    }
    func_refcounts[0] = u32::MAX;
    func_refcounts
}
