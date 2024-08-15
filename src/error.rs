/*
 * libpathrs: safe path resolution on Linux
 * Copyright (C) 2019-2021 Aleksa Sarai <cyphar@cyphar.com>
 * Copyright (C) 2019-2021 SUSE LLC
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

//! Error types for libpathrs.

// NOTE: This module is mostly a workaround until several issues have been
//       resolved:
//
//  * `std::error::Error::chain` is stabilised.
//  * I figure out a nice way to implement GlobalBacktrace...

use crate::{resolvers::opath::SymlinkStackError, syscalls::Error as SyscallError};

use std::{borrow::Cow, error::Error as StdError, io::Error as IOError};

// TODO: Add a backtrace to Error. We would just need to add an automatic
//       Backtrace::capture() in From. But it's not clear whether we want to
//       export the crate types here without std::backtrace::Backtrace.

#[derive(thiserror::Error, Debug)]
#[error(transparent)]
pub struct Error(#[from] Box<ErrorImpl>);

impl From<ErrorImpl> for Error {
    fn from(err: ErrorImpl) -> Self {
        Self(Box::new(err))
    }
}

impl Error {
    pub(crate) fn kind(&self) -> ErrorKind {
        self.0.kind()
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum ErrorImpl {
    #[error("feature {feature} is not implemented")]
    NotImplemented { feature: Cow<'static, str> },

    #[error("feature {feature} not supported on this kernel")]
    NotSupported { feature: Cow<'static, str> },

    #[error("invalid {name} argument: {description}")]
    InvalidArgument {
        name: Cow<'static, str>,
        description: Cow<'static, str>,
    },

    #[error("violation of safety requirement: {description}")]
    SafetyViolation { description: Cow<'static, str> },

    #[error("broken symlink stack during iteration: {description}")]
    BadSymlinkStackError {
        description: Cow<'static, str>,
        source: SymlinkStackError,
    },

    #[error("{operation} failed")]
    OsError {
        operation: Cow<'static, str>,
        source: IOError,
    },

    #[error("{operation} failed")]
    RawOsError {
        operation: Cow<'static, str>,
        source: SyscallError,
    },

    #[error("{context}")]
    Wrapped {
        context: Cow<'static, str>,
        source: Box<ErrorImpl>,
    },
}

// TODO: Export this?
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[non_exhaustive]
pub(crate) enum ErrorKind {
    NotImplemented,
    NotSupported,
    InvalidArgument,
    SafetyViolation,
    BadSymlinkStack,
    // TODO: We might want to use Option<std::io::ErrorKind>?
    OsError(Option<i32>),
}

impl ErrorImpl {
    pub(crate) fn kind(&self) -> ErrorKind {
        match self {
            Self::NotImplemented { .. } => ErrorKind::NotImplemented,
            Self::NotSupported { .. } => ErrorKind::NotSupported,
            Self::InvalidArgument { .. } => ErrorKind::InvalidArgument,
            Self::SafetyViolation { .. } => ErrorKind::SafetyViolation,
            Self::BadSymlinkStackError { .. } => ErrorKind::BadSymlinkStack,
            Self::OsError { source, .. } => ErrorKind::OsError(source.raw_os_error()),
            Self::RawOsError { source, .. } => {
                ErrorKind::OsError(source.root_cause().raw_os_error())
            }
            Self::Wrapped { source, .. } => source.kind(),
        }
    }
}

// Private trait necessary to work around the "orphan trait" restriction.
pub(crate) trait ErrorExt: Sized {
    /// Wrap a `Result<..., Error>` with an additional context string.
    fn wrap<S: Into<String>>(self, context: S) -> Self {
        self.with_wrap(|| context.into())
    }

    /// Wrap a `Result<..., Error>` with an additional context string created by
    /// a closure.
    fn with_wrap<F>(self, context_fn: F) -> Self
    where
        F: FnOnce() -> String;
}

impl ErrorExt for ErrorImpl {
    fn with_wrap<F>(self, context_fn: F) -> Self
    where
        F: FnOnce() -> String,
    {
        Self::Wrapped {
            context: context_fn().into(),
            source: self.into(),
        }
    }
}

impl ErrorExt for Error {
    fn with_wrap<F>(self, context_fn: F) -> Self
    where
        F: FnOnce() -> String,
    {
        self.0.with_wrap(context_fn).into()
    }
}

impl<T, E: ErrorExt> ErrorExt for Result<T, E> {
    fn with_wrap<F>(self, context_fn: F) -> Self
    where
        F: FnOnce() -> String,
    {
        self.map_err(|err| err.with_wrap(context_fn))
    }
}

/// A backport of the nightly-only [`Chain`]. This method
/// will be removed as soon as that is stabilised.
///
/// [`Chain`]: https://doc.rust-lang.org/nightly/std/error/struct.Chain.html
// XXX: https://github.com/rust-lang/rust/issues/58520
pub(crate) struct Chain<'a> {
    current: Option<&'a (dyn StdError + 'static)>,
}

impl<'a> Iterator for Chain<'a> {
    type Item = &'a (dyn StdError + 'static);

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;
        self.current = self.current.and_then(StdError::source);
        current
    }
}

impl Error {
    /// A backport of the nightly-only [`Error::chain`]. This method
    /// will be removed as soon as that is stabilised.
    ///
    /// [`Error::chain`]: https://doc.rust-lang.org/nightly/std/error/trait.Error.html#method.chain
    // XXX: https://github.com/rust-lang/rust/issues/58520
    pub(crate) fn iter_chain_hotfix(&self) -> Chain {
        Chain {
            current: Some(self),
        }
    }
}
