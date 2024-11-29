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

// We need to permit unsafe code because we are interacting with libc APIs.
#![allow(unsafe_code)]

use crate::{
    flags::{OpenFlags, RenameFlags},
    utils::{FdExt, ToCString},
};

use std::{
    ffi::OsStr,
    fmt,
    io::Error as IOError,
    mem::MaybeUninit,
    os::unix::{
        ffi::OsStrExt,
        io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd},
    },
    path::{Path, PathBuf},
    sync::LazyLock,
};

use rustix::{
    fs::{
        self as rustix_fs, AtFlags, Dev, FileType, Mode, RawMode, Stat, StatFs, Statx, StatxFlags,
    },
    io::Errno,
    mount::{self as rustix_mount, FsMountFlags, FsOpenFlags, MountAttrFlags, OpenTreeFlags},
    process as rustix_process, thread as rustix_thread,
};

// TODO: Figure out how we can put a backtrace here (it seems we can't use
//       thiserror's backtrace support without nightly Rust because thiserror
//       wants to be able to derive an Error for Backtrace?). We could add a
//       backtrace to error::Error but if we also add a backtrace to
//       syscalls::Error this might get a little complicated.
// MSRV(1.65): Use std::backtrace::Backtrace.

// SAFETY: AT_FDCWD is always a valid file descriptor.
pub(crate) const AT_FDCWD: BorrowedFd<'static> = rustix_fs::CWD;
// SAFETY: BADFD is not a valid file descriptor, but it's not -1.
#[cfg(test)]
pub(crate) const BADFD: BorrowedFd<'static> = unsafe { BorrowedFd::borrow_raw(-libc::EBADF) };

/// Representation of a file descriptor and its associated path at a given point
/// in time.
///
/// This is primarily used to make pretty-printing syscall arguments much nicer,
/// and users really shouldn't be interacting with this directly.
///
/// # Caveats
/// Note that the file descriptor value is very unlikely to reference a live
/// file descriptor. Its value is only used for informational purposes.
#[derive(Clone, Debug)]
pub(crate) struct FrozenFd(RawFd, Option<PathBuf>);

// TODO: Should probably be a pub(crate) impl.
impl<Fd: AsFd> From<Fd> for FrozenFd {
    fn from(fd: Fd) -> Self {
        // SAFETY: as_unsafe_path is safe here since it is only used for
        //         pretty-printing error messages and no real logic.
        Self(fd.as_fd().as_raw_fd(), fd.as_unsafe_path_unchecked().ok())
    }
}

impl fmt::Display for FrozenFd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            libc::AT_FDCWD => write!(f, "[AT_FDCWD]")?,
            fd => write!(f, "[{}]", fd)?,
        };
        match &self.1 {
            Some(path) => write!(f, "{path:?}")?,
            None => write!(f, "<unknown>")?,
        };
        Ok(())
    }
}

/// Internal error returned by libpathrs's syscall wrappers.
///
/// The primary thing of note is that these errors contain detailed debugging
/// information about the arguments to each given syscall. Users would most
/// often not interact with these error variants directly and instead would make
/// use of the top-level [`Error`] type.
///
/// [`Error`]: crate::error::Error
#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {
    // NOTE: This is temporary until the issue is fixed in rustix.
    #[error("invalid file descriptor {fd} (see <https://github.com/bytecodealliance/rustix/issues/1187> for more details)")]
    InvalidFd { fd: RawFd, source: Errno },

    #[error("openat({dirfd}, {path}, {flags:?}, 0o{mode:o})")]
    Openat {
        dirfd: FrozenFd,
        path: PathBuf,
        flags: OpenFlags,
        mode: u32,
        source: Errno,
    },

    #[error("openat2({dirfd}, {path}, {how}, {size})")]
    Openat2 {
        dirfd: FrozenFd,
        path: PathBuf,
        how: OpenHow,
        size: usize,
        source: Errno,
    },

    #[error("readlinkat({dirfd}, {path})")]
    Readlinkat {
        dirfd: FrozenFd,
        path: PathBuf,
        source: Errno,
    },

    #[error("mkdirat({dirfd}, {path}, 0o{mode:o})")]
    Mkdirat {
        dirfd: FrozenFd,
        path: PathBuf,
        mode: u32,
        source: Errno,
    },

    #[error("mknodat({dirfd}, {path}, 0o{mode:o}, {major}:{minor})")]
    Mknodat {
        dirfd: FrozenFd,
        path: PathBuf,
        mode: u32,
        major: u32,
        minor: u32,
        source: Errno,
    },

    #[error("unlinkat({dirfd}, {path}, {flags:?})")]
    Unlinkat {
        dirfd: FrozenFd,
        path: PathBuf,
        flags: AtFlags,
        source: Errno,
    },

    #[error("linkat({old_dirfd}, {old_path}, {new_dirfd}, {new_path}, {flags:?})")]
    Linkat {
        old_dirfd: FrozenFd,
        old_path: PathBuf,
        new_dirfd: FrozenFd,
        new_path: PathBuf,
        flags: AtFlags,
        source: Errno,
    },

    #[error("symlinkat({dirfd}, {path}, {target})")]
    Symlinkat {
        dirfd: FrozenFd,
        path: PathBuf,
        target: PathBuf,
        source: Errno,
    },

    #[error("renameat({old_dirfd}, {old_path}, {new_dirfd}, {new_path})")]
    Renameat {
        old_dirfd: FrozenFd,
        old_path: PathBuf,
        new_dirfd: FrozenFd,
        new_path: PathBuf,
        source: Errno,
    },

    #[error("renameat2({old_dirfd}, {old_path}, {new_dirfd}, {new_path}, {flags:?})")]
    Renameat2 {
        old_dirfd: FrozenFd,
        old_path: PathBuf,
        new_dirfd: FrozenFd,
        new_path: PathBuf,
        flags: RenameFlags,
        source: Errno,
    },

    #[error("fstatfs({fd})")]
    Fstatfs { fd: FrozenFd, source: Errno },

    #[error("fstatat({dirfd}, {path}, 0x{flags:x})")]
    Fstatat {
        dirfd: FrozenFd,
        path: PathBuf,
        flags: AtFlags,
        source: Errno,
    },

    #[error("statx({dirfd}, {path}, flags=0x{flags:x}, mask=0x{mask:x})")]
    Statx {
        dirfd: FrozenFd,
        path: PathBuf,
        flags: AtFlags,
        mask: StatxFlags,
        source: Errno,
    },

    #[error("fsopen({fstype}, {flags:?})")]
    Fsopen {
        fstype: String,
        flags: FsOpenFlags,
        source: Errno,
    },

    #[error("fsconfig({sfd}, FSCONFIG_CMD_CREATE)")]
    FsconfigCreate { sfd: FrozenFd, source: Errno },

    #[error("fsconfig({sfd}, FSCONFIG_SET_STRING, {key:?}, {value:?})")]
    FsconfigSetString {
        sfd: FrozenFd,
        key: String,
        value: String,
        source: Errno,
    },

    #[error("fsmount({sfd}, {flags:?}, {mount_attrs:?})")]
    Fsmount {
        sfd: FrozenFd,
        flags: FsMountFlags,
        mount_attrs: MountAttrFlags,
        source: Errno,
    },

    #[error("open_tree({dirfd}, {path}, {flags:?})")]
    OpenTree {
        dirfd: FrozenFd,
        path: PathBuf,
        flags: OpenTreeFlags,
        source: Errno,
    },
}

impl Error {
    fn errno(&self) -> &Errno {
        // XXX: This should probably be a macro...
        match self {
            Error::InvalidFd { source, .. } => source,
            Error::Openat { source, .. } => source,
            Error::Openat2 { source, .. } => source,
            Error::Readlinkat { source, .. } => source,
            Error::Mkdirat { source, .. } => source,
            Error::Mknodat { source, .. } => source,
            Error::Unlinkat { source, .. } => source,
            Error::Linkat { source, .. } => source,
            Error::Symlinkat { source, .. } => source,
            Error::Renameat { source, .. } => source,
            Error::Renameat2 { source, .. } => source,
            Error::Fstatfs { source, .. } => source,
            Error::Fstatat { source, .. } => source,
            Error::Statx { source, .. } => source,
            Error::Fsopen { source, .. } => source,
            Error::FsconfigCreate { source, .. } => source,
            Error::FsconfigSetString { source, .. } => source,
            Error::Fsmount { source, .. } => source,
            Error::OpenTree { source, .. } => source,
        }
    }

    // TODO: Switch to returning &Errno.
    pub(crate) fn root_cause(&self) -> IOError {
        IOError::from_raw_os_error(self.errno().raw_os_error())
    }
}

// FIXME: Temporary fix for rustix panicking if a passed file descriptor is
// non-standard <https://github.com/bytecodealliance/rustix/issues/1187>.
trait HotfixRustixFd: Sized {
    fn hotfix_rustix_fd(self) -> Result<Self, Error>;
}

impl<Fd: AsFd + Sized> HotfixRustixFd for Fd {
    fn hotfix_rustix_fd(self) -> Result<Self, Error> {
        match self.as_fd().as_raw_fd() {
            libc::AT_FDCWD | 0.. => Ok(self),
            fd => Err(Error::InvalidFd {
                fd,
                source: Errno::BADF,
            }),
        }
    }
}

/// Wrapper for `openat(2)` which auto-sets `O_CLOEXEC | O_NOCTTY`.
///
/// This is needed because Rust doesn't provide a way to access the dirfd
/// argument of `openat(2)`. We need the dirfd argument, so we need a wrapper.
pub(crate) fn openat_follow<Fd: AsFd, P: AsRef<Path>>(
    dirfd: Fd,
    path: P,
    mut flags: OpenFlags,
    mode: RawMode, // TODO: Should we take rustix::fs::Mode directly?
) -> Result<OwnedFd, Error> {
    let dirfd = dirfd.as_fd().hotfix_rustix_fd()?;
    let path = path.as_ref();

    // O_CLOEXEC is needed for obvious reasons, and O_NOCTTY ensures that a
    // malicious file won't take control of our terminal.
    flags.insert(OpenFlags::O_CLOEXEC | OpenFlags::O_NOCTTY);

    rustix_fs::openat(dirfd, path, flags.into(), Mode::from_raw_mode(mode)).map_err(|errno| {
        Error::Openat {
            dirfd: dirfd.into(),
            path: path.into(),
            flags,
            mode,
            source: errno,
        }
    })
}

/// Wrapper for `openat(2)` which auto-sets `O_CLOEXEC | O_NOCTTY | O_NOFOLLOW`.
///
/// This is needed because Rust doesn't provide a way to access the dirfd
/// argument of `openat(2)`. We need the dirfd argument, so we need a wrapper.
pub(crate) fn openat<Fd: AsFd, P: AsRef<Path>>(
    dirfd: Fd,
    path: P,
    mut flags: OpenFlags,
    mode: RawMode, // TODO: Should we take rustix::fs::Mode directly?
) -> Result<OwnedFd, Error> {
    flags.insert(OpenFlags::O_NOFOLLOW);
    openat_follow(dirfd, path, flags, mode)
}

/// Wrapper for `readlinkat(2)`.
///
/// This is needed because Rust doesn't provide a way to access the dirfd
/// argument of `readlinkat(2)`. We need the dirfd argument, so we need a
/// wrapper.
pub(crate) fn readlinkat<Fd: AsFd, P: AsRef<Path>>(dirfd: Fd, path: P) -> Result<PathBuf, Error> {
    let dirfd = dirfd.as_fd().hotfix_rustix_fd()?;
    let path = path.as_ref();

    // If the contents of the symlink are larger than this, we bail out avoid
    // DoS vectors (because there is no way to get the size of a symlink
    // beforehand, you just have to read it).
    // MSRV(1.79): Use const {}?
    let mut linkbuf: [MaybeUninit<u8>; 32 * 4096] =
        [MaybeUninit::uninit(); 32 * libc::PATH_MAX as usize];

    let (target, trailing) =
        rustix_fs::readlinkat_raw(dirfd, path, &mut linkbuf[..]).map_err(|errno| {
            Error::Readlinkat {
                dirfd: dirfd.into(),
                path: path.into(),
                source: errno,
            }
        })?;

    if trailing.is_empty() {
        // The buffer was too small, return an error.
        Err(Error::Readlinkat {
            dirfd: dirfd.into(),
            path: path.into(),
            source: Errno::NAMETOOLONG,
        })
    } else {
        Ok(PathBuf::from(OsStr::from_bytes(target)))
    }
}

/// Wrapper for `mkdirat(2)`.
///
/// This is needed because Rust doesn't provide a way to access the dirfd
/// argument of `mkdirat(2)`. We need the dirfd argument, so we need a wrapper.
pub(crate) fn mkdirat<Fd: AsFd, P: AsRef<Path>>(
    dirfd: Fd,
    path: P,
    mode: RawMode, // TODO: Should we take rustix::fs::Mode directly?
) -> Result<(), Error> {
    let dirfd = dirfd.as_fd().hotfix_rustix_fd()?;
    let path = path.as_ref();

    rustix_fs::mkdirat(dirfd, path, Mode::from_raw_mode(mode)).map_err(|errno| Error::Mkdirat {
        dirfd: dirfd.into(),
        path: path.into(),
        mode,
        source: errno,
    })
}

pub(crate) fn devmajorminor(dev: Dev) -> (u32, u32) {
    (rustix_fs::major(dev), rustix_fs::minor(dev))
}

/// Wrapper for `mknodat(2)`.
///
/// This is needed because Rust doesn't provide a way to access the dirfd
/// argument of `mknodat(2)`. We need the dirfd argument, so we need a wrapper.
pub(crate) fn mknodat<Fd: AsFd, P: AsRef<Path>>(
    dirfd: Fd,
    path: P,
    raw_mode: RawMode, // TODO: Should we take rustix::fs::{Mode,FileType} directly?
    dev: Dev,
) -> Result<(), Error> {
    let dirfd = dirfd.as_fd().hotfix_rustix_fd()?;
    let path = path.as_ref();
    let (file_type, mode) = (
        FileType::from_raw_mode(raw_mode),
        Mode::from_raw_mode(raw_mode),
    );

    rustix_fs::mknodat(dirfd, path, file_type, mode, dev).map_err(|errno| {
        let (major, minor) = devmajorminor(dev);
        Error::Mknodat {
            dirfd: dirfd.into(),
            path: path.into(),
            mode: raw_mode,
            major,
            minor,
            source: errno,
        }
    })
}

/// Wrapper for `unlinkat(2)`.
///
/// This is needed because Rust doesn't provide a way to access the dirfd
/// argument of `unlinkat(2)`. We need the dirfd argument, so we need a wrapper.
pub(crate) fn unlinkat<Fd: AsFd, P: AsRef<Path>>(
    dirfd: Fd,
    path: P,
    flags: AtFlags,
) -> Result<(), Error> {
    let dirfd = dirfd.as_fd().hotfix_rustix_fd()?;
    let path = path.as_ref();

    rustix_fs::unlinkat(dirfd, path, flags).map_err(|errno| Error::Unlinkat {
        dirfd: dirfd.into(),
        path: path.into(),
        flags,
        source: errno,
    })
}

/// Wrapper for `linkat(2)`.
///
/// This is needed because Rust doesn't provide a way to access the dirfd
/// argument of `linkat(2)`. We need the dirfd argument, so we need a wrapper.
pub(crate) fn linkat<Fd1: AsFd, P1: AsRef<Path>, Fd2: AsFd, P2: AsRef<Path>>(
    old_dirfd: Fd1,
    old_path: P1,
    new_dirfd: Fd2,
    new_path: P2,
    flags: AtFlags,
) -> Result<(), Error> {
    let (old_dirfd, old_path) = (old_dirfd.as_fd().hotfix_rustix_fd()?, old_path.as_ref());
    let (new_dirfd, new_path) = (new_dirfd.as_fd().hotfix_rustix_fd()?, new_path.as_ref());

    rustix_fs::linkat(old_dirfd, old_path, new_dirfd, new_path, flags).map_err(|errno| {
        Error::Linkat {
            old_dirfd: old_dirfd.into(),
            old_path: old_path.into(),
            new_dirfd: new_dirfd.into(),
            new_path: new_path.into(),
            flags,
            source: errno,
        }
    })
}

/// Wrapper for `symlinkat(2)`.
///
/// This is needed because Rust doesn't provide a way to access the dirfd
/// argument of `symlinkat(2)`. We need the dirfd argument, so we need a
/// wrapper.
pub(crate) fn symlinkat<P1: AsRef<Path>, Fd: AsFd, P2: AsRef<Path>>(
    target: P1,
    dirfd: Fd,
    path: P2,
) -> Result<(), Error> {
    let (dirfd, path) = (dirfd.as_fd().hotfix_rustix_fd()?, path.as_ref());
    let target = target.as_ref();

    rustix_fs::symlinkat(target, dirfd, path).map_err(|errno| Error::Symlinkat {
        dirfd: dirfd.into(),
        path: path.into(),
        target: target.into(),
        source: errno,
    })
}

/// Wrapper for `renameat(2)`.
///
/// This is needed because Rust doesn't provide a way to access the dirfd
/// argument of `renameat(2)`. We need the dirfd argument, so we need a wrapper.
pub(crate) fn renameat<Fd1: AsFd, P1: AsRef<Path>, Fd2: AsFd, P2: AsRef<Path>>(
    old_dirfd: Fd1,
    old_path: P1,
    new_dirfd: Fd2,
    new_path: P2,
) -> Result<(), Error> {
    let (old_dirfd, old_path) = (old_dirfd.as_fd().hotfix_rustix_fd()?, old_path.as_ref());
    let (new_dirfd, new_path) = (new_dirfd.as_fd().hotfix_rustix_fd()?, new_path.as_ref());

    rustix_fs::renameat(old_dirfd, old_path, new_dirfd, new_path).map_err(|errno| Error::Renameat {
        old_dirfd: old_dirfd.into(),
        old_path: old_path.into(),
        new_dirfd: new_dirfd.into(),
        new_path: new_path.into(),
        source: errno,
    })
}

pub(crate) static RENAME_FLAGS_SUPPORTED: LazyLock<bool> = LazyLock::new(|| {
    match renameat2(AT_FDCWD, ".", AT_FDCWD, ".", RenameFlags::RENAME_EXCHANGE) {
        Ok(_) => true,
        // We expect EBUSY, but just to be safe we only check for ENOSYS.
        Err(err) => err.root_cause().raw_os_error() != Some(libc::ENOSYS),
    }
});

/// Wrapper for `renameat2(2)`.
///
/// This is needed because Rust doesn't provide any interface for `renameat2(2)`
/// (especially not an interface for the dirfd).
pub(crate) fn renameat2<Fd1: AsFd, P1: AsRef<Path>, Fd2: AsFd, P2: AsRef<Path>>(
    old_dirfd: Fd1,
    old_path: P1,
    new_dirfd: Fd2,
    new_path: P2,
    flags: RenameFlags,
) -> Result<(), Error> {
    // Use renameat(2) if no flags are specified.
    if flags.is_empty() {
        return renameat(old_dirfd, old_path, new_dirfd, new_path);
    }

    let (old_dirfd, old_path) = (old_dirfd.as_fd().hotfix_rustix_fd()?, old_path.as_ref());
    let (new_dirfd, new_path) = (new_dirfd.as_fd().hotfix_rustix_fd()?, new_path.as_ref());

    rustix_fs::renameat_with(old_dirfd, old_path, new_dirfd, new_path, flags.into()).map_err(
        |errno| Error::Renameat2 {
            old_dirfd: old_dirfd.into(),
            old_path: old_path.into(),
            new_dirfd: new_dirfd.into(),
            new_path: new_path.into(),
            flags,
            source: errno,
        },
    )
}

/// Wrapper for `fstatfs(2)`.
///
/// This is needed because Rust doesn't provide any interface for `fstatfs(2)`.
pub(crate) fn fstatfs<Fd: AsFd>(fd: Fd) -> Result<StatFs, Error> {
    let fd = fd.as_fd().hotfix_rustix_fd()?;

    rustix_fs::fstatfs(fd).map_err(|errno| Error::Fstatfs {
        fd: fd.into(),
        source: errno,
    })
}

/// Wrapper for `fstatat(2)`, which auto-sets `AT_NO_AUTOMOUNT |
/// AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH`.
///
/// This is needed because Rust doesn't provide any interface for `fstatat(2)`.
pub(crate) fn fstatat<Fd: AsFd, P: AsRef<Path>>(dirfd: Fd, path: P) -> Result<Stat, Error> {
    let dirfd = dirfd.as_fd().hotfix_rustix_fd()?;
    let path = path.as_ref();
    let flags = AtFlags::NO_AUTOMOUNT | AtFlags::SYMLINK_NOFOLLOW | AtFlags::EMPTY_PATH;

    rustix_fs::statat(dirfd, path, flags).map_err(|errno| Error::Fstatat {
        dirfd: dirfd.into(),
        path: path.into(),
        flags,
        source: errno,
    })
}

pub(crate) fn statx<Fd: AsFd, P: AsRef<Path>>(
    dirfd: Fd,
    path: P,
    mask: StatxFlags,
) -> Result<Statx, Error> {
    let dirfd = dirfd.as_fd().hotfix_rustix_fd()?;
    let path = path.as_ref();
    let flags = AtFlags::NO_AUTOMOUNT | AtFlags::SYMLINK_NOFOLLOW | AtFlags::EMPTY_PATH;

    rustix_fs::statx(dirfd, path, flags, mask).map_err(|errno| Error::Statx {
        dirfd: dirfd.into(),
        path: path.into(),
        flags,
        mask,
        source: errno,
    })
}

pub(crate) static OPENAT2_IS_SUPPORTED: LazyLock<bool> =
    LazyLock::new(|| openat2(AT_FDCWD, ".", &Default::default()).is_ok());

bitflags! {
    /// Wrapper for the underlying `libc`'s `RESOLVE_*` flags.
    ///
    /// The flag values and their meaning is identical to the description in the
    /// [`openat2(2)`] man page.
    ///
    /// [`openat2(2)`]: http://man7.org/linux/man-pages/man2/openat2.2.html
    #[derive(Default, PartialEq, Eq, Debug, Clone, Copy)]
    struct ResolveFlags: u64 {
        const RESOLVE_BENEATH = libc::RESOLVE_BENEATH;
        const RESOLVE_IN_ROOT = libc::RESOLVE_IN_ROOT;
        const RESOLVE_NO_MAGICLINKS = libc::RESOLVE_NO_MAGICLINKS;
        const RESOLVE_NO_SYMLINKS = libc::RESOLVE_NO_SYMLINKS;
        const RESOLVE_NO_XDEV = libc::RESOLVE_NO_XDEV;
        const RESOLVE_CACHED = libc::RESOLVE_CACHED;

        // Don't clobber unknown RESOLVE_* bits.
        const _ = !0;
    }
}

/// Arguments for how `openat2` should open the target path.
#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct OpenHow {
    /// O_* flags (`-EINVAL` on unknown or incompatible flags).
    pub flags: u64,
    /// O_CREAT or O_TMPFILE file mode (must be zero otherwise).
    pub mode: u64,
    /// RESOLVE_* flags (`-EINVAL` on unknown flags).
    pub resolve: u64,
}

impl fmt::Display for OpenHow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ ")?;
        // self.flags
        if let Ok(oflags) = i32::try_from(self.flags) {
            // If the flags can fit inside OpenFlags, pretty-print the flags.
            write!(f, "flags: {:?}, ", OpenFlags::from_bits_retain(oflags))?;
        } else {
            write!(f, "flags: 0x{:x}, ", self.flags)?;
        }
        if self.flags & (libc::O_CREAT | libc::O_TMPFILE) as u64 != 0 {
            write!(f, "mode: 0o{:o}, ", self.mode)?;
        }
        // self.resolve
        write!(
            f,
            "resolve: {:?}",
            ResolveFlags::from_bits_retain(self.resolve)
        )?;
        write!(f, " }}")
    }
}

// NOTE: rustix's openat2 wrapper is not extensible-friendly so we use our own
// for now. See <https://github.com/bytecodealliance/rustix/issues/1186>.
pub(crate) fn openat2<Fd: AsFd, P: AsRef<Path>>(
    dirfd: Fd,
    path: P,
    how: &OpenHow,
) -> Result<OwnedFd, Error> {
    let dirfd = dirfd.as_fd().hotfix_rustix_fd()?;
    let path = path.as_ref();

    // Add O_CLOEXEC explicitly. No need for O_NOFOLLOW because
    // RESOLVE_IN_ROOT handles that correctly in a race-free way.
    let mut how = how.clone();
    how.flags |= libc::O_CLOEXEC as u64;

    // SAFETY: Obviously safe-to-use Linux syscall.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            dirfd.as_raw_fd(),
            path.to_c_string().as_ptr(),
            &how as *const OpenHow,
            std::mem::size_of::<OpenHow>(),
        )
    } as RawFd;
    let err = IOError::last_os_error();

    if fd >= 0 {
        // SAFETY: We know it's a real file descriptor.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    } else {
        Err(Error::Openat2 {
            dirfd: dirfd.into(),
            path: path.into(),
            how,
            size: std::mem::size_of::<OpenHow>(),
            source: err
                .raw_os_error()
                .map(Errno::from_raw_os_error)
                .expect("syscall failure must result in a real OS error"),
        })
    }
}

#[cfg(test)]
pub(crate) fn getpid() -> rustix_process::RawPid {
    rustix_process::Pid::as_raw(Some(rustix_process::getpid()))
}

pub(crate) fn gettid() -> rustix_process::RawPid {
    rustix_process::Pid::as_raw(Some(rustix_thread::gettid()))
}

pub(crate) fn geteuid() -> rustix_process::RawUid {
    rustix_process::geteuid().as_raw()
}

#[cfg(test)]
pub(crate) fn getegid() -> rustix_process::RawGid {
    rustix_process::getegid().as_raw()
}

#[cfg(test)]
pub(crate) fn getcwd() -> Result<PathBuf, anyhow::Error> {
    let buffer = Vec::with_capacity(libc::PATH_MAX as usize);
    Ok(OsStr::from_bytes(rustix_process::getcwd(buffer)?.to_bytes()).into())
}

pub(crate) fn fsopen<S: AsRef<str>>(fstype: S, flags: FsOpenFlags) -> Result<OwnedFd, Error> {
    let fstype = fstype.as_ref();

    rustix_mount::fsopen(fstype, flags).map_err(|errno| Error::Fsopen {
        fstype: fstype.into(),
        flags,
        source: errno,
    })
}

pub(crate) fn fsconfig_set_string<Fd: AsFd, K: AsRef<str>, V: AsRef<str>>(
    sfd: Fd,
    key: K,
    value: V,
) -> Result<(), Error> {
    let sfd = sfd.as_fd().hotfix_rustix_fd()?;
    let key = key.as_ref();
    let value = value.as_ref();

    rustix_mount::fsconfig_set_string(sfd, key, value).map_err(|errno| Error::FsconfigSetString {
        sfd: sfd.into(),
        key: key.into(),
        value: value.into(),
        source: errno,
    })
}

pub(crate) fn fsconfig_create<Fd: AsFd>(sfd: Fd) -> Result<(), Error> {
    let sfd = sfd.as_fd().hotfix_rustix_fd()?;

    rustix_mount::fsconfig_create(sfd).map_err(|errno| Error::FsconfigCreate {
        sfd: sfd.into(),
        source: errno,
    })
}

pub(crate) fn fsmount<Fd: AsFd>(
    sfd: Fd,
    flags: FsMountFlags,
    mount_attrs: MountAttrFlags,
) -> Result<OwnedFd, Error> {
    let sfd = sfd.as_fd().hotfix_rustix_fd()?;

    rustix_mount::fsmount(sfd, flags, mount_attrs).map_err(|errno| Error::Fsmount {
        sfd: sfd.into(),
        flags,
        mount_attrs,
        source: errno,
    })
}

pub(crate) fn open_tree<Fd: AsFd, P: AsRef<Path>>(
    dirfd: Fd,
    path: P,
    flags: OpenTreeFlags,
) -> Result<OwnedFd, Error> {
    let dirfd = dirfd.as_fd().hotfix_rustix_fd()?;
    let path = path.as_ref();

    rustix_mount::open_tree(dirfd, path, flags).map_err(|errno| Error::OpenTree {
        dirfd: dirfd.into(),
        path: path.into(),
        flags,
        source: errno,
    })
}
