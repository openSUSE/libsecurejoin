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

//! libpathrs::opath implements an emulated version of openat2(RESOLVE_IN_ROOT).
//! The primary method by which this is done is through shameless abuse of
//! procfs and O_PATH magic-links. The basic idea is that we need to perform all
//! of the path resolution steps (walking down the set of components, handling
//! the effect of symlinks on the resolution, etc).
//!
//! In order to do this safely we need to verify after the walk is done whether
//! the path of the final file descriptor is what we expected (most importantly,
//! is it inside the root which we started the walk with?). This check is done
//! through readlink(/proc/self/fd/$n), which is a magic kernel interface which
//! gives you the kernel's view of the path -- and in cases where the kernel is
//! unsure or otherwise unhappy you get "/".
//!
//! If the check fails, we assume we are being attacked and return an error (and
//! the caller can decide to re-try if they want). The kernel implementation
//! will fail in fewer cases because it has access to in-kernel locks and other
//! measures, but the final check through procfs should block all attack
//! attempts.

use crate::{
    error::{Error, ErrorExt, ErrorImpl},
    flags::{OpenFlags, ResolverFlags},
    procfs::GLOBAL_PROCFS_HANDLE,
    resolvers::{opath::SymlinkStack, PartialLookup, MAX_SYMLINK_TRAVERSALS},
    syscalls,
    utils::{self, FdExt, PathIterExt},
    Handle,
};

use std::{
    collections::VecDeque,
    ffi::{OsStr, OsString},
    io::Error as IOError,
    iter,
    os::unix::{
        ffi::OsStrExt,
        fs::MetadataExt,
        io::{AsFd, OwnedFd},
    },
    path::{Path, PathBuf},
    rc::Rc,
    sync::LazyLock,
};

use itertools::Itertools;

/// Ensure that the expected path within the root matches the current fd.
fn check_current<RootFd: AsFd, Fd: AsFd, P: AsRef<Path>>(
    current: Fd,
    root: RootFd,
    expected: P,
) -> Result<(), Error> {
    // SAFETY: as_unsafe_path is safe here since we're using it to build a path
    //         for a string-based check as part of a larger safety setup. This
    //         path will be re-checked after the unsafe "current_path" is
    //         generated.
    let root_path = root
        .as_unsafe_path(&GLOBAL_PROCFS_HANDLE)
        .wrap("get root path to construct expected path")?;

    // Combine the root path and our expected_path to get the full path to
    // compare current against.
    let full_path: PathBuf = root_path.join(
        // Path::join() has the unfortunate behaviour that a leading "/" will
        // result in the prefix path being removed. In practice we don't ever
        // hit this case (probably because RawComponents doesn't explicitly have
        // an equivalent of Components::RootDir), but just to be sure prepend a
        // "." component anyway.
        iter::once(OsStr::from_bytes(b"."))
            .chain(expected.as_ref().raw_components())
            // NOTE: PathBuf::push() does not normalise components.
            .collect::<PathBuf>(),
    );

    // Does /proc/self/fd agree with us? There are several circumstances where
    // this check might give a false positive (namely, if the kernel decides
    // that the path is not ordinarily resolveable). But if this check passes,
    // then we can be fairly sure (barring kernel bugs) that the path was safe
    // at least one point in time.
    // SAFETY: as_unsafe_path is safe here since we're explicitly doing a
    //         string-based check to see whether the path we want is correct.
    let current_path = current
        .as_unsafe_path(&GLOBAL_PROCFS_HANDLE)
        .wrap("check fd against expected path")?;

    // The paths should be identical.
    if current_path != full_path {
        Err(ErrorImpl::SafetyViolation {
            description: format!(
                "fd doesn't match expected path ({} != {})",
                current_path.display(),
                full_path.display()
            )
            .into(),
        })?
    }

    // And the root should not have moved. Note that this check could (in
    // theory) be bypassed by an attacker -- so it important that users be aware
    // that allowing roots to be moved by an attacker is a very bad idea.
    // SAFETY: as_unsafe_path path is safe here because it's just used in a
    //         string check -- and it's known that this check isn't perfect.
    let new_root_path = root
        .as_unsafe_path(&GLOBAL_PROCFS_HANDLE)
        .wrap("get root path to double-check it hasn't moved")?;
    if root_path != new_root_path {
        Err(ErrorImpl::SafetyViolation {
            description: "root moved during lookup".into(),
        })?
    }

    Ok(())
}

/// Cached copy of `fs.protected_symlinks` sysctl.
// TODO: In theory this value could change during the lifetime of the
// program, but there's no nice way of detecting that, and the overhead of
// checking this for every symlink lookup is more likely to be an issue.
static PROTECTED_SYMLINKS_SYSCTL: LazyLock<u32> = LazyLock::new(|| {
    utils::sysctl_read_parse(&GLOBAL_PROCFS_HANDLE, "fs.protected_symlinks")
        .expect("should be able to parse fs.protected_symlinks")
});

/// Verify that we should follow the symlink as per `fs.protected_symlinks`.
///
/// Because we emulate symlink following in userspace, the kernel cannot apply
/// `fs.protected_symlinks` restrictions so we need to emulate them ourselves.
fn may_follow_link<DirFd: AsFd, Fd: AsFd>(dir: DirFd, link: Fd) -> Result<(), Error> {
    // Skip doing checks if the fs.protected_symlinks sysctl is disabled.
    let fsuid = syscalls::geteuid();
    let dir_meta = dir.metadata().wrap("fetch directory metadata")?;
    let link_meta = link.metadata().wrap("fetch symlink metadata")?;

    const STICKY_WRITABLE: libc::mode_t = libc::S_ISVTX | libc::S_IWOTH;

    // We only do this if fs.protected_symlinks is enabled.
    if *PROTECTED_SYMLINKS_SYSCTL == 0 ||
        // Allowed if owner and follower match.
        link_meta.uid() == fsuid ||
        // Allowed if the directory is not sticky and world-writable.
        dir_meta.mode() & STICKY_WRITABLE != STICKY_WRITABLE ||
        // Allowed if parent directory and link owner match.
        link_meta.uid() == dir_meta.uid()
    {
        Ok(())
    } else {
        Err(ErrorImpl::OsError {
            operation: "emulated fs.protected_symlinks".into(),
            source: IOError::from_raw_os_error(libc::EACCES),
        }
        .into())
    }
}

/// Common implementation used by `resolve_partial()` and `resolve()`. The main
/// difference is that if `symlink_stack` is `true`, the returned paths
// TODO: Make (flags, no_follow_trailing, symlink_stack) a single struct to
//       avoid possible issues with passing a bool to the wrong argument.
fn do_resolve<Fd: AsFd, P: AsRef<Path>>(
    root: Fd,
    path: P,
    flags: ResolverFlags,
    no_follow_trailing: bool,
    mut symlink_stack: Option<&mut SymlinkStack<OwnedFd>>,
) -> Result<PartialLookup<Rc<OwnedFd>>, Error> {
    // What is the final path we expect to get after we do the final open? This
    // allows us to track any attacker moving path components around and we can
    // sanity-check at the very end. This does not include rootpath.
    let mut expected_path = PathBuf::from("/");

    // We only need to keep track of our current dirfd, since we are applying
    // the components one-by-one, and can always switch back to the root
    // if we hit an absolute symlink.
    let root = Rc::new(
        root.as_fd()
            .try_clone_to_owned()
            .map_err(|err| ErrorImpl::OsError {
                operation: "dup root handle as starting point of resolution".into(),
                source: err,
            })?,
    );
    let mut current = Rc::clone(&root);

    // Get initial set of components from the passed path. We remove components
    // as we do the path walk, and update them with the contents of any symlinks
    // we encounter. Path walking terminates when there are no components left.
    let mut remaining_components = path
        .raw_components()
        .map(|p| p.to_os_string())
        .collect::<VecDeque<_>>();

    let mut symlink_traversals = 0;
    while let Some(part) = remaining_components.pop_front() {
        // Stash a copy of the real remaining path. We can't just use
        // ::collect<PathBuf> because we might have "" components, which
        // std::path::PathBuf don't like.
        let remaining: PathBuf = Itertools::intersperse(
            iter::once(&part)
                .chain(remaining_components.iter())
                .map(OsString::as_os_str),
            OsStr::new("/"),
        )
        .collect::<OsString>()
        .into();

        let part = match part.as_bytes() {
            // If we hit an empty component, we need to treat it as though it is
            // "." so that trailing "/" and "//" components on a non-directory
            // correctly return the right error code.
            b"" => ".".into(),
            // For "." component we don't touch expected_path, but we do try to
            // do the open (to return the correct openat2-compliant error if the
            // current path is a not directory).
            b"." => part,
            b".." => {
                // All of expected_path is non-symlinks, so we can treat ".."
                // lexically. If pop() fails, then we are at the root.
                // should .
                if !expected_path.pop() {
                    current = Rc::clone(&root);
                    continue;
                }
                part
            }
            _ => {
                // This part might be a symlink, but we clean that up later.
                expected_path.push(&part);

                // Ensure that part doesn't contain any "/"s. It's critical we
                // are only touching the final component in the path. If there
                // are any other path components we must bail. This shouldn't
                // ever happen, but it's better to be safe.
                if part.as_bytes().contains(&b'/') {
                    Err(ErrorImpl::SafetyViolation {
                        description: "component of path resolution contains '/'".into(),
                    })?
                }

                part
            }
        };

        // Get our next element.
        // MSRV(1.69): Remove &*.
        match syscalls::openat(
            &*current,
            &part,
            OpenFlags::O_PATH | OpenFlags::O_NOFOLLOW,
            0,
        )
        .map_err(|err| {
            ErrorImpl::RawOsError {
                operation: "open next component of resolution".into(),
                source: err,
            }
            .into()
        }) {
            Err(err) => {
                return Ok(PartialLookup::Partial {
                    handle: current,
                    remaining,
                    last_error: err,
                });
            }
            Ok(next) => {
                // Make sure that the path is what we expect. If not, there was
                // a racing rename and we should bail out here -- otherwise we
                // might be tricked into revealing information outside the
                // rootfs through error or timing-related attacks.
                //
                // The safety argument for only needing to check ".." is
                // identical to the kernel implementation (namely, walking down
                // is safe by-definition). However, unlike the in-kernel version
                // we don't have the luxury of only doing this check when there
                // was a racing rename -- we have to do it every time.
                if part.as_bytes() == b".." {
                    // MSRV(1.69): Remove &*.
                    check_current(&next, &*root, &expected_path)
                        .wrap("check next '..' component didn't escape")?;
                }

                // Is the next dirfd a symlink or an ordinary path? If we're an
                // ordinary dirent, we just update current and move on to the
                // next component. Nothing special here.
                if !next
                    .metadata()
                    .wrap("fstat of next component")?
                    .is_symlink()
                {
                    // We hit a non-symlink component, so clear it from the
                    // symlink stack.
                    if let Some(ref mut stack) = symlink_stack {
                        stack
                            .pop_part(&part)
                            .map_err(|err| ErrorImpl::BadSymlinkStackError {
                                description: "walking into component".into(),
                                source: err,
                            })?;
                    }
                    // Just keep walking.
                    current = next.into();
                    continue;
                } else {
                    // If we hit the last component and we were told to not follow
                    // the trailing symlink, just return the link we have.
                    if remaining_components.is_empty() && no_follow_trailing {
                        current = next.into();
                        break;
                    }

                    // Don't continue walking if user asked for no symlinks.
                    if flags.contains(ResolverFlags::NO_SYMLINKS) {
                        return Ok(PartialLookup::Partial {
                            handle: current,
                            remaining,
                            // Construct a fake OS error containing ELOOP.
                            last_error: ErrorImpl::OsError {
                                operation: "emulated symlink resolution".into(),
                                source: IOError::from_raw_os_error(libc::ELOOP),
                            }
                            .wrap(format!(
                                "component {part:?} is a symlink but symlink resolution is disabled",
                            ))
                            .into(),
                        });
                    }

                    // Verify that we can follow the link.
                    // MSRV(1.69): Remove &*.
                    may_follow_link(&*current, &next).with_wrap(|| {
                        format!(
                            "component {part:?} is an unsafe symlink that is blocked by fs.protected_symlinks"
                        )
                    })?;

                    // We need a limit on the number of symlinks we traverse to
                    // avoid hitting filesystem loops and DoSing.
                    symlink_traversals += 1;
                    if symlink_traversals >= MAX_SYMLINK_TRAVERSALS {
                        return Ok(PartialLookup::Partial {
                            handle: current,
                            remaining,
                            // Construct a fake OS error containing ELOOP.
                            last_error: ErrorImpl::OsError {
                                operation: "emulated symlink resolution".into(),
                                source: IOError::from_raw_os_error(libc::ELOOP),
                            }
                            .wrap("exceeded symlink limit")
                            .into(),
                        });
                    }

                    let link_target =
                        syscalls::readlinkat(&next, "").map_err(|err| ErrorImpl::RawOsError {
                            operation: "readlink next symlink component".into(),
                            source: err,
                        })?;

                    // Check if it's a good idea to walk this symlink. If we are on
                    // a filesystem that supports magic-links and we've hit an
                    // absolute symlink, it is incredibly likely that this component
                    // is a magic-link and it makes no sense to try to resolve it in
                    // userspace.
                    //
                    // NOTE: There are some pseudo-magic-links like /proc/self
                    // (which dynamically generates the symlink contents but doesn't
                    // use nd_jump_link). In the case of procfs, these are always
                    // relative, and they are reasonable for us to walk.
                    //
                    // In procfs, all magic-links use d_path() to generate
                    // readlink() and thus are all absolute paths. (Unfortunately,
                    // apparmorfs uses nd_jump_link to make
                    // /sys/kernel/security/apparmor/policy dynamic using actual
                    // nd_jump_link() and their readlink give us a dummy relative
                    // path like "apparmorfs:[123]". But in that case we will just
                    // get an error.)
                    if link_target.is_absolute()
                        && next
                            .is_magiclink_filesystem()
                            .wrap("check if next is on a dangerous filesystem")?
                    {
                        Err(ErrorImpl::OsError {
                            operation: "emulated RESOLVE_NO_MAGICLINKS".into(),
                            source: IOError::from_raw_os_error(libc::ELOOP),
                        })
                        .wrap("walked into a potential magic-link")?
                    }

                    // Swap out the symlink component in the symlink stack with
                    // a new entry for the link target.
                    if let Some(ref mut stack) = symlink_stack {
                        stack
                            .swap_link(&part, (&current, remaining), link_target.clone())
                            .map_err(|err| ErrorImpl::BadSymlinkStackError {
                                description: "walking into symlink".into(),
                                source: err,
                            })?;
                    }

                    // Remove the link component from our expectex path.
                    expected_path.pop();

                    // Add contents of the symlink to the set of components we are
                    // looping over.
                    link_target
                        .raw_components()
                        .prepend(&mut remaining_components);

                    // Absolute symlinks reset our current state back to /.
                    if link_target.is_absolute() {
                        current = Rc::clone(&root);
                        expected_path = PathBuf::from("/");
                    }
                }
            }
        }
    }

    // Make sure that the path is what we expect...
    // MSRV(1.69): Remove &*.
    check_current(&*current, &*root, &expected_path).wrap("check final handle didn't escape")?;

    // We finished the lookup with no remaining components.
    Ok(PartialLookup::Complete(current))
}

/// Resolve as many components as possible in `path` within `root` through
/// user-space emulation.
pub(crate) fn resolve_partial<Fd: AsFd, P: AsRef<Path>>(
    root: Fd,
    path: P,
    flags: ResolverFlags,
    no_follow_trailing: bool,
) -> Result<PartialLookup<Rc<OwnedFd>>, Error> {
    // For partial lookups, we need to use a SymlinkStack to match openat2.
    let mut symlink_stack = SymlinkStack::new();

    match do_resolve(
        root,
        path,
        flags,
        no_follow_trailing,
        Some(&mut symlink_stack),
    ) {
        // For complete and error paths, just return what we got.
        ret @ Ok(PartialLookup::Complete(_)) => ret,
        err @ Err(_) => err,

        // If the lookup failed part-way through, modify the (handle, remaining)
        // based on the symlink stack if applicable.
        Ok(PartialLookup::Partial {
            handle,
            remaining,
            last_error,
        }) => match symlink_stack.pop_top_symlink() {
            // We were in the middle of symlink resolution, so return the error
            // from the context of the top symlink in the resolution, to match
            // openat2(2).
            Some((handle, remaining)) => Ok(PartialLookup::Partial {
                handle,
                remaining,
                last_error,
            }),
            // Nothing in the symlink stack, return what we got.
            None => Ok(PartialLookup::Partial {
                handle,
                remaining,
                last_error,
            }),
        },
    }
}

/// Resolve `path` within `root` through user-space emulation.
pub(crate) fn resolve<Fd: AsFd, P: AsRef<Path>>(
    root: Fd,
    path: P,
    flags: ResolverFlags,
    no_follow_trailing: bool,
) -> Result<Handle, Error> {
    do_resolve(root, path, flags, no_follow_trailing, None).and_then(TryInto::try_into)
}
