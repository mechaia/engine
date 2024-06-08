use {
    crate::{
        program::{Constant, Function, Instruction},
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

        let Function::Block(b) = &program.functions[i] else {
            continue;
        };

        for &instr in b.instructions.iter() {
            match instr {
                Instruction::Call { address } if should_inline[address as usize] => {
                    // TODO avoid inlining infinite loop function

                    let f_2 = &program.functions[address as usize];
                    let Function::Block(b_2) = f_2 else { todo!() };
                    new_instrs.extend_from_slice(&b_2.instructions);

                    if let Some(next_2) = b_2.next {
                        new_instrs.push(Instruction::Call { address });
                    }
                }
                instr => new_instrs.push(instr),
            }
        }

        let Function::Block(b) = &mut program.functions[i] else {
            unreachable!()
        };
        b.instructions = new_instrs.into();
    }

    sort_pre_order(program);
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

fn visit_call_tree_rec(
    new_list: &mut Vec<u32>,
    program: &Program,
    index: u32,
    visited: &mut BitVec,
) {
}
