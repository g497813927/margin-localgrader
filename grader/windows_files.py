"""Small Windows file boundary for settings-only persistence.

Windows has no dir_fd/O_NOFOLLOW equivalent in Python's os API. Pin the config
directory against rename and open files as reparse points before inspecting them.
This module is imported only on Windows; no platform dependency is needed.
"""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import os


@contextmanager
def checked_handle(path, *, directory=False):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   wintypes.LPVOID, wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL

    class AttributeInfo(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("reparse_tag", wintypes.DWORD)]

    class FileIdInfo(ctypes.Structure):
        _fields_ = [("volume", ctypes.c_ulonglong), ("file_id", ctypes.c_ubyte * 16)]

    def file_identity(open_handle):
        identity = FileIdInfo()
        if not kernel.GetFileInformationByHandleEx(open_handle, 18, ctypes.byref(identity),
                                                   ctypes.sizeof(identity)):
            raise OSError("Workspace directory identity could not be verified.")
        file_id = bytes(identity.file_id)
        if not any(file_id):
            raise OSError("Workspace directory has no stable file identity.")
        return identity.volume, file_id

    # Hold the directory handle and verify that its original pathname still
    # refers to the same object. Windows can rename an open directory even
    # when its handle does not share delete access.
    handle = kernel.CreateFileW(str(path), 0x80 if directory else 0x80000000,
                                3, None, 3,
                                0x00200000 | (0x02000000 if directory else 0), None)
    if handle == ctypes.c_void_p(-1).value:
        code = ctypes.get_last_error()
        if code in (2, 3):
            raise FileNotFoundError("Workspace settings do not exist.")
        raise OSError("Workspace settings could not be opened safely.")
    try:
        info = AttributeInfo()
        if not kernel.GetFileInformationByHandleEx(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError("Workspace attributes could not be verified.")
        if info.attributes & 0x400 or bool(info.attributes & 0x10) != directory:
            raise ValueError("Unsafe workspace file or directory.")
        pinned_identity = file_identity(handle) if directory else None
        yield handle
        if directory:
            current = kernel.CreateFileW(str(path), 0x80, 3, None, 3,
                                         0x00200000 | 0x02000000, None)
            if current == ctypes.c_void_p(-1).value:
                raise OSError("Workspace directory changed during access.")
            try:
                if file_identity(current) != pinned_identity:
                    raise OSError("Workspace directory changed during access.")
            finally:
                kernel.CloseHandle(current)
    finally:
        kernel.CloseHandle(handle)


def read_settings(path, maximum):
    import msvcrt
    import stat
    with checked_handle(path) as handle:
        # Duplicate ownership for fdopen so the context manager closes only its
        # own handle. os.dup retains the already verified file, never a pathname.
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.DuplicateHandle.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                          wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE),
                                          wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.DuplicateHandle.restype = wintypes.BOOL
        process = kernel.GetCurrentProcess()
        duplicate = wintypes.HANDLE()
        if not kernel.DuplicateHandle(process, handle, process, ctypes.byref(duplicate), 0, False, 2):
            raise OSError("Workspace settings handle could not be read.")
        descriptor = msvcrt.open_osfhandle(duplicate.value, os.O_RDONLY | os.O_BINARY)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("Unsafe workspace file.")
            return stream.read(maximum + 1)
