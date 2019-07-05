/*
 * libpathrs: safe path resolution on Linux
 * Copyright (C) 2019 Aleksa Sarai <cyphar@cyphar.com>
 * Copyright (C) 2019 SUSE LLC
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use crate::syscalls::unstable;

use crate::{Handle, Root};

use core::convert::TryFrom;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use failure::{Error as FailureError, ResultExt};

lazy_static! {
    pub static ref IS_SUPPORTED: bool = {
        let how = unstable::OpenHow::new();
        unstable::openat2(libc::AT_FDCWD, ".", &how).is_ok()
    };
}

/// Resolve `path` within `root` through `openat2(2)`.
pub fn resolve<P: AsRef<Path>>(root: &Root, path: P) -> Result<Handle, FailureError> {
    if !*IS_SUPPORTED {
        bail!("kernel resolution is not supported on this kernel")
    }

    let mut how = unstable::OpenHow::new();
    how.flags = libc::O_PATH;
    how.resolve = unstable::RESOLVE_IN_ROOT;
    let file = unstable::openat2(root.as_raw_fd(), path, &how).context("open sub-path")?;

    let handle = Handle::try_from(file).context("convert RESOLVE_IN_ROOT fd to Handle")?;
    Ok(handle)
}
