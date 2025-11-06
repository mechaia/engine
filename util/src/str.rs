use {
    crate::{Arena, ArenaHandle},
    core::fmt,
    std::{num::NonZeroUsize, ops, sync::Mutex},
};

static POOL: Mutex<Pool> = Mutex::new(Pool {
    strings: Arena::new(),
    // TODO fast mapping from string to handle
});

struct Pool {
    strings: Arena<(Box<[u8]>, NonZeroUsize)>,
}

/// A minimally-sized string of UTF-8 characters.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PoolStr(PoolBoxU8);

/// A minimally-sized string of bytes.
///
/// Strings are stored in a global pool and referenced by an arena handle.
/// This both reduces memory usage if many strings are duplicated and
/// enables much faster comparisons between any two `PoolStr`.
///
/// To keep comparisons fast, `PartialOrd` looks only at the index and not the string contents.
/// This means `PoolStr` will *not* be lexicographically ordered.
#[derive(PartialEq, Eq, Hash)]
pub struct PoolBoxU8(ArenaHandle);

impl Pool {
    fn get_and_own(&mut self, s: &[u8]) -> Option<ArenaHandle> {
        self.strings
            .iter_mut()
            .find(|(_, v)| &*v.0 == s)
            .map(|(h, v)| {
                v.1 = v.1.saturating_add(1);
                h
            })
    }

    fn add(&mut self, s: Box<[u8]>) -> ArenaHandle {
        self.strings.insert((s, NonZeroUsize::MIN))
    }

    fn drop_owner(&mut self, h: ArenaHandle) {
        let s = &mut self.strings[h];
        if let Some(refc) = NonZeroUsize::new(s.1.get() - 1) {
            s.1 = refc;
        } else {
            self.strings.remove(h);
        }
    }

    fn add_owner(&mut self, h: ArenaHandle) {
        let v = &mut self.strings[h];
        v.1 = v.1.saturating_add(1);
    }
}

impl From<&str> for PoolStr {
    fn from(s: &str) -> Self {
        Self(PoolBoxU8::from(s.as_bytes()))
    }
}

impl From<Box<str>> for PoolStr {
    fn from(s: Box<str>) -> Self {
        Self(PoolBoxU8::from(s.into_boxed_bytes()))
    }
}

impl From<String> for PoolStr {
    fn from(s: String) -> Self {
        Self(PoolBoxU8::from(s.into_bytes()))
    }
}

impl From<PoolStr> for PoolBoxU8 {
    fn from(s: PoolStr) -> Self {
        s.0
    }
}

impl From<&[u8]> for PoolBoxU8 {
    fn from(s: &[u8]) -> Self {
        let mut p = POOL.lock().unwrap();
        Self(p.get_and_own(s).unwrap_or_else(|| p.add(s.into())))
    }
}

impl From<&str> for PoolBoxU8 {
    fn from(s: &str) -> Self {
        PoolStr::from(s).0
    }
}

impl From<Box<[u8]>> for PoolBoxU8 {
    fn from(s: Box<[u8]>) -> Self {
        let mut p = POOL.lock().unwrap();
        Self(p.get_and_own(&s).unwrap_or_else(|| p.add(s)))
    }
}

impl From<Vec<u8>> for PoolBoxU8 {
    fn from(s: Vec<u8>) -> Self {
        let mut p = POOL.lock().unwrap();
        Self(p.get_and_own(&s).unwrap_or_else(|| p.add(s.into())))
    }
}

impl From<&PoolBoxU8> for Vec<u8> {
    fn from(value: &PoolBoxU8) -> Self {
        POOL.lock().unwrap().strings[value.0].0.clone().into()
    }
}

impl Clone for PoolBoxU8 {
    fn clone(&self) -> Self {
        POOL.lock().unwrap().add_owner(self.0);
        Self(self.0)
    }
}

impl Drop for PoolBoxU8 {
    fn drop(&mut self) {
        POOL.lock().unwrap().drop_owner(self.0)
    }
}

impl ops::Add<&PoolBoxU8> for &str {
    type Output = PoolBoxU8;

    fn add(self, rhs: &PoolBoxU8) -> Self::Output {
        let p = POOL.lock().unwrap();
        let r = &p.strings[rhs.0].0;
        PoolBoxU8::from(self.bytes().chain(r.iter().copied()).collect::<Vec<_>>())
    }
}

impl ops::Add<&PoolBoxU8> for &PoolBoxU8 {
    type Output = PoolBoxU8;

    fn add(self, rhs: &PoolBoxU8) -> Self::Output {
        let p = POOL.lock().unwrap();
        let l = &p.strings[self.0].0;
        let r = &p.strings[rhs.0].0;
        PoolBoxU8::from(l.iter().chain(r.iter()).copied().collect::<Vec<_>>())
    }
}

impl fmt::Debug for PoolBoxU8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = &*POOL.lock().unwrap().strings[self.0].0;
        loop {
            match core::str::from_utf8(s) {
                Ok(s) => {
                    s.fmt(f)?;
                    break;
                }
                Err(e) => {
                    let (a, b) = s.split_at(e.valid_up_to());
                    a.fmt(f)?;
                    f.write_str(" ")?;
                    b[0].fmt(f)?;
                    s = &b[1..];
                    f.write_str(" ")?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Debug for PoolStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
