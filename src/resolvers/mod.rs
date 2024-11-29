/*
 * libpathrs: safe path resolution on Linux
 * Copyright (C) 2019-2024 Aleksa Sarai <cyphar@cyphar.com>
 * Copyright (C) 2019-2024 SUSE LLC
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#![forbid(unsafe_code)]

//! Resolver implementations for libpathrs.

use crate::{
    error::{Error, ErrorKind},
    flags::ResolverFlags,
    syscalls, Handle,
};

use std::{
    os::unix::io::{AsFd, OwnedFd},
    path::{Path, PathBuf},
    rc::Rc,
    sync::LazyLock,
};

/// `O_PATH`-based userspace resolver.
pub(crate) mod opath;
/// `openat2(2)`-based in-kernel resolver.
pub(crate) mod openat2;
/// A limited resolver only used for `/proc` lookups in `ProcfsHandle`.
pub(crate) mod procfs;

/// Maximum number of symlink traversals we will accept.
const MAX_SYMLINK_TRAVERSALS: usize = 128;

/// The backend used for path resolution within a [`Root`] to get a [`Handle`].
///
/// We don't generally recommend specifying this, since libpathrs will
/// automatically detect the best backend for your platform (which is the value
/// returned by `Resolver::default`). However, this can be useful for testing.
///
/// [`Root`]: crate::Root
/// [`Handle`]: crate::Handle
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub(crate) enum ResolverBackend {
    /// Use the native `openat2(2)` backend (requires kernel support).
    KernelOpenat2,
    /// Use the userspace "emulated" backend.
    EmulatedOpath,
    // TODO: Implement a HardcoreEmulated which does pivot_root(2) and all the
    //       rest of it. It'd be useful to compare against and for some
    //       hyper-concerned users.
}

static DEFAULT_RESOLVER_TYPE: LazyLock<ResolverBackend> = LazyLock::new(|| {
    if *syscalls::OPENAT2_IS_SUPPORTED {
        ResolverBackend::KernelOpenat2
    } else {
        ResolverBackend::EmulatedOpath
    }
});

impl Default for ResolverBackend {
    fn default() -> Self {
        *DEFAULT_RESOLVER_TYPE
    }
}

impl ResolverBackend {
    /// Checks if the resolver is supported on the current platform.
    #[cfg(test)]
    pub(crate) fn supported(self) -> bool {
        match self {
            ResolverBackend::KernelOpenat2 => *syscalls::OPENAT2_IS_SUPPORTED,
            ResolverBackend::EmulatedOpath => true,
        }
    }
}

/// Resolover backend and its associated flags.
///
/// This is the primary structure used to configure how a given [`Root`] will
/// conduct path resolutions.
///
/// [`Root`]: crate::Root
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Resolver {
    /// Underlying resolution backend used.
    pub(crate) backend: ResolverBackend,
    /// Flags to pass to the resolution backend.
    pub flags: ResolverFlags,
}

/// Only used for internal resolver implementations.
#[derive(Debug)]
pub(crate) enum PartialLookup<H, E = Error> {
    Complete(H),
    Partial {
        handle: H,
        remaining: PathBuf,
        last_error: E,
    },
}

impl<H> AsRef<H> for PartialLookup<H> {
    fn as_ref(&self) -> &H {
        match self {
            Self::Complete(handle) => handle,
            Self::Partial { handle, .. } => handle,
        }
    }
}

impl TryInto<Handle> for PartialLookup<Handle> {
    type Error = Error;

    fn try_into(self) -> Result<Handle, Self::Error> {
        match self {
            Self::Complete(handle) => Ok(handle),
            Self::Partial { last_error, .. } => Err(last_error),
        }
    }
}

impl TryInto<Handle> for PartialLookup<Rc<OwnedFd>> {
    type Error = Error;

    fn try_into(self) -> Result<Handle, Self::Error> {
        PartialLookup::<Handle>::from(self).try_into()
    }
}

impl TryInto<(Handle, Option<PathBuf>)> for PartialLookup<Handle> {
    type Error = Error;

    fn try_into(self) -> Result<(Handle, Option<PathBuf>), Self::Error> {
        match self {
            Self::Complete(handle) => Ok((handle, None)),
            Self::Partial {
                handle,
                remaining,
                last_error,
            } => match last_error.kind() {
                ErrorKind::OsError(Some(libc::ENOENT)) => Ok((handle, Some(remaining))),
                _ => Err(last_error),
            },
        }
    }
}

impl TryInto<(Handle, Option<PathBuf>)> for PartialLookup<Rc<OwnedFd>> {
    type Error = Error;

    fn try_into(self) -> Result<(Handle, Option<PathBuf>), Self::Error> {
        PartialLookup::<Handle>::from(self).try_into()
    }
}

impl From<PartialLookup<Rc<OwnedFd>>> for PartialLookup<Handle> {
    fn from(result: PartialLookup<Rc<OwnedFd>>) -> Self {
        let (rc, partial) = match result {
            PartialLookup::Complete(rc) => (rc, None),
            PartialLookup::Partial {
                handle,
                remaining,
                last_error,
            } => (handle, Some((remaining, last_error))),
        };

        // We are now sure that there is only a single reference to whatever
        // current points to. There is nowhere else we could've stashed a
        // reference, and we only do Rc::clone for root (which we've dropped).
        let handle = Handle::from_fd(
            // MSRV(1.70): Use Rc::into_inner().
            Rc::try_unwrap(rc)
                .expect("current handle in lookup should only have a single Rc reference"),
        );

        match partial {
            None => Self::Complete(handle),
            Some((remaining, last_error)) => Self::Partial {
                handle,
                remaining,
                last_error,
            },
        }
    }
}

impl Resolver {
    /// Internal dispatcher to the relevant backend.
    #[inline]
    pub(crate) fn resolve<F: AsFd, P: AsRef<Path>>(
        &self,
        root: F,
        path: P,
        no_follow_trailing: bool,
    ) -> Result<Handle, Error> {
        match self.backend {
            ResolverBackend::KernelOpenat2 => {
                openat2::resolve(root, path, self.flags, no_follow_trailing)
            }
            ResolverBackend::EmulatedOpath => {
                opath::resolve(root, path, self.flags, no_follow_trailing)
            }
        }
    }

    #[inline]
    pub(crate) fn resolve_partial<F: AsFd, P: AsRef<Path>>(
        &self,
        root: F,
        path: P,
        no_follow_trailing: bool,
    ) -> Result<PartialLookup<Handle>, Error> {
        match self.backend {
            ResolverBackend::KernelOpenat2 => {
                openat2::resolve_partial(root, path.as_ref(), self.flags, no_follow_trailing)
            }
            ResolverBackend::EmulatedOpath => {
                opath::resolve_partial(root, path.as_ref(), self.flags, no_follow_trailing)
                    // Rc<File> -> Handle
                    .map(Into::into)
            }
        }
    }
}
