"""
This effectively  walks through module import statements recursively to find
all modules that a given one depends on.
It furthermore manages the packaging of newly found dependencies when requested

ISSUES: For speed, this does not use pathhooks unless imp.find_module fails.
Consequently, if modules can be found in two different sys.path entries,
the order
processed by this module may differ from the python import system
Entirely arbitrary pathhooks are not supported for now - only ZipImporter
    (specifically importers with a archive attribute)

There are some hacks to deal with transmitting archives -- we coerce archives
to be stored
to cloud.archives/archive.
An eventual goal is to clean up the hackish pathhook support code

Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.
All rights reserved.

email: contact@picloud.com

The cloud package is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this package; if not, see
http://www.gnu.org/licenses/lgpl-2.1.html
"""

from __future__ import with_statement
import os
import sys
import modulefinder
import imp
import marshal
import dis

#from ..serialization import cloudpickle
import cloudpickle
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s,%(msecs)03d  %(levelname)-8s  \
                        %(module)-16s  L:%(lineno)03d  P:%(process)-4d  \
                        T:%(thread)-4d  %(message)s",
                    datefmt='%H:%M:%S')
cloudLog = logging.getLogger(__name__)

#from .. import cloudconfig as cc
#cloudLog = logging.getLogger("Cloud.Transport")

LOAD_CONST = chr(dis.opname.index('LOAD_CONST'))
IMPORT_NAME = chr(dis.opname.index('IMPORT_NAME'))
STORE_NAME = chr(dis.opname.index('STORE_NAME'))
STORE_GLOBAL = chr(dis.opname.index('STORE_GLOBAL'))
STORE_OPS = [STORE_NAME, STORE_GLOBAL]
HAVE_ARGUMENT = chr(dis.HAVE_ARGUMENT)

ZIP_IMPORT = -9  # custom imp-like type code


class DependencyManager(modulefinder.ModuleFinder):
    """
    Based off of module finder.

    Features:
    -IgnoreList to ignore base python packages for performance purposes
    -Timestamp tracking
    -'Snapshots' to determine new modules
    -Warnings on using custom C extensions

    Note: This is not thread safe: The user of this is responsible for
    locking it down

    TODO: Be smart with using import hooks (get_code)
    """

    @staticmethod
    def formatIgnoreList(unformattedList):
        """Format the ignore list"""

        builtins = sys.builtin_module_names

        ignoreList = {}
        for module in unformattedList:
            modname = module.strip()
            if modname[0] == '#' or modname[0] == ';':
                continue
            modname = modname.split('.')

            if modname[-1] == '*':
                ignoreList[tuple(modname[:-1])] = True
            else:
                # set to False if .* isn't required
                ignoreList[tuple(modname)] = True

        #add builtins:
        for builtin in builtins:
            ignoreList[(builtin, )] = True

        return ignoreList

    def __init__(self, path=sys.path, debug=0, excludes=[], replace_paths=[]):
        modulefinder.ModuleFinder.__init__(self, path, debug, None,
                                           replace_paths)
        self.ignoreList = self.formatIgnoreList(excludes)
        self.lastSnapshot = set()  # tracking
        self.transitError = set()  # avoid excessive extension warnings

        # analyze main which is not transmitted
        m = sys.modules['__main__']
        if hasattr(m, '__file__') and cloudpickle.useForcedImports:
            self.inject_module(m)
            # if main imports a.b we might not see that b has been loaded
            # The below is a hack to detect this case:
            checkModules = self.modules.keys() + self.get_ignored_modules()
            for mod in checkModules:
                self.msgout(2, "inspect", m)
                if '.' in mod:
                    loadedmod = sys.modules.get(mod)
                    # if mod is not loaded, import is triggered in a function:
                    if loadedmod:
                        if not hasattr(m, '___pyc_forcedImports__'):
                            m.___pyc_forcedImports__ = set()
                        m.___pyc_forcedImports__.add(loadedmod)

    def shouldIgnore(self, modname):
        """Check ignoreList to determine if this module should
            not be processed"""
        modname = tuple(modname.split('.'))

        if modname in self.ignoreList:
            return True
        for i in range(1, len(modname)):
            tst = modname[0:-i]
            # print 'doing test', tst
            val = self.ignoreList.get(tst)
            if val:  # Must be true
                return True
        return False

    def load_package(self, fqname, pathname, archive_name=None):
        """Fix bug with not passing parent into find_module"""
        self.msgin(2, "load_package", fqname, pathname)
        newname = modulefinder.replacePackageMap.get(fqname)
        if newname:
            fqname = newname

        if archive_name:  # part of an archive
            m = self.add_module(fqname,
                                filename=archive_name,
                                path=[pathname] + \
                                modulefinder.packagePathMap.get(fqname,
                                                                []),
                                is_archive=True)
        else:
            # As per comment in modulefinder, simulate runtime __path__
            # additions.
            m = self.add_module(fqname,
                                filename=pathname + '/__init__.py',
                                path=[pathname] + \
                                    modulefinder.packagePathMap.get(fqname,
                                                                    []))

        # Bug fix.  python2.6 modulefinder doesn't pass parent to find_module
        fp, buf, stuff = self.find_module("__init__", m.__path__, parent=m)

        self.load_module(fqname, fp, buf, stuff)
        self.msgout(2, "load_package ->", m)
        return m

    def inject_module(self, mod):
        """High level module adding.
        This adds an actual module from sys.modules into the finder
        """

        mname = mod.__name__
        if mname in self.modules:
            return
        if self.shouldIgnore(mname):
            return

        if mname == '__main__':  # special case
            searchnames = []
            dirs, mod = os.path.split(mod.__file__)
            searchname = mod.split('.', 1)[0]  # extract fully qualified name
        else:
            searchnames = mname.rsplit('.', 1)
            # must load parents first...
            pkg = searchnames[0]

            if len(searchnames) > 1 and pkg not in self.modules:
                self.inject_module(sys.modules[pkg])

            searchname = searchnames[-1]
        if len(searchnames) > 1:  # this module has a parent - resolve it
            path = sys.modules[pkg].__path__
        else:
            path = None

        try:
            fp, pathname, stuff = self.find_module(searchname, path)
            self.load_module(mname, fp, pathname, stuff)
        except ImportError:  # Ignore import errors
            pass

    def add_module(self, fqname, filename, path=None, is_archive=False):
        """Save timestamp here"""
        if fqname in self.modules:
            return self.modules[fqname]
        # print 'pre-adding %s' % fqname
        if not filename:  # ignore any builtin or extension
            return

        if is_archive:
            # module's filename is set to the actual archive
            relfilename = os.path.split(filename)[1]
            # cloudpickle needs to know about this to deserialize correctly:
        else:
            # get relative path:
            numsplits = fqname.count('.') + 1
            relfilename = ""
            absfilename = filename
            for i in xrange(numsplits):
                absfilename, tmp = os.path.split(absfilename)
                relfilename = tmp + '/' + relfilename
                if '__init__' in tmp:
                    # additional split as this is a package and
                    # __init__ is not in fqname
                    absfilename, tmp = os.path.split(absfilename)
                    relfilename = tmp + '/' + relfilename
            relfilename = relfilename[:-1]  # remove terminating /

        self.modules[fqname] = m = modulefinder.Module(fqname,
                                                       relfilename, path)
        # picloud: Timestamp module for update checks
        m.timestamp = long(os.path.getmtime(filename))
        m.is_archive = is_archive
        return m

    """Manually try to find name on sys.path_hooks
    Some code taken from python3.1 implib"""
    def _path_hooks(self, path):
        """Search path hooks for a finder for 'path'.
        """
        hooks = sys.path_hooks
        for hook in hooks:
            try:
                finder = hook(path)
                sys.path_importer_cache[path] = finder
                return finder
            except ImportError:
                continue
        return None

    def manual_find(self, name, path):
        """Load with pathhooks. Return none if fails to load or if default
        importer must be used
        Otherwise returns loader object, path_loader_handles"""
        finder = None
        for entry in path:
            try:
                finder = sys.path_importer_cache[entry]
            except KeyError:
                finder = self._path_hooks(entry)
            if finder:
                loader = finder.find_module(name)
                if loader:
                    return loader, entry
        return None, None  # nothing found!

    def find_module(self, name, path, parent=None):
        """find_module using ignoreList
        TODO: Somehow use pathhooks here
        """
        if parent is not None:
            # assert path is not None
            fullname = parent.__name__ + '.' + name
        else:
            fullname = name
        # print 'test to ignore %s -- %s -- %s' % (fullname, parent, path)

        if self.shouldIgnore(fullname):
            self.msgout(3, "find_module -> Ignored", fullname)
            # PEP8 CHANGE
            raise ImportError(name)

        if path is None:
            if name in sys.builtin_module_names:
                return (None, None, ("", "", imp.C_BUILTIN))

            path = sys.path
        # print 'imp is scanning for %s at %s' % (name, path)
        try:
            return imp.find_module(name, path)
        except ImportError:
            # try path hooks
            loader, ldpath = self.manual_find(name, path)
            if not loader:
                raise
            #We now have a PEP 302 loader object. Internally, we must format it

            if not hasattr(loader, 'archive') or not hasattr(loader,
                                                             'get_code'):
                if fullname not in self.transitError:
                    cloudLog.warn("Cloud cannot transmit python module '%s'.\
                    It needs to be imported by a %s path hook, but such a path\
                    hook \
                    does not provide both the \
                    'archive' and 'get_code' property..  Import errors may\
                    result;\
                    please see PiCloud documentation." % (fullname,
                                                          str(loader)))
                    self.transitError.add(fullname)
                raise

            return (None,  ldpath + '/' + name, (loader, name, ZIP_IMPORT))

    def get_ignored_modules(self):
        """Return list of modules that are used but were ignored"""
        ignored = []
        for name in self.badmodules:
            if self.shouldIgnore(name):
                ignored.append(name)
        return ignored

    def any_missing_maybe(self):
        """Return two lists, one with modules that are certainly missing
        and one with modules that *may* be missing. The latter names could
        either be submodules *or* just global names in the package.

        The reason it can't always be determined is that it's impossible to
        tell which names are imported when "from module import *" is done
        with an extension module, short of actually importing it.

        PiCloud: Use ignoreList
        """
        missing = []
        maybe = []
        for name in self.badmodules:
            if self.shouldIgnore(name):
                continue
            i = name.rfind(".")
            if i < 0:
                missing.append(name)
                continue
            subname = name[i + 1:]
            pkgname = name[:i]
            pkg = self.modules.get(pkgname)
            if pkg is not None:
                if pkgname in self.badmodules[name]:
                    # The package tried to import this module itself and
                    # failed. It's definitely missing.
                    missing.append(name)
                elif subname in pkg.globalnames:
                    # It's a global in the package: definitely not missing.
                    pass
                elif pkg.starimports:
                    # It could be missing, but the package did an "import *"
                    # from a non-Python module, so we simply can't be sure.
                    maybe.append(name)
                else:
                    # It's not a global in the package, the package didn't
                    # do funny star imports, it's very likely to be missing.
                    # The symbol could be inserted into the package from the
                    # outside, but since that's not good style we simply list
                    # it missing.
                    missing.append(name)
            else:
                missing.append(name)
        missing.sort()
        maybe.sort()
        return missing, maybe

    def load_module(self, fqname, fp, pathname, file_info):
        suffix, mode, type = file_info
        #PiCloud: Warn on C extensions and __import_
        self.msgin(2, "load_module", fqname, fp and "fp", pathname)
        if type == ZIP_IMPORT:
            #archive (as suffix) is an PEP 302 importer that implements
            # archive and get_code
            #pathname is used to access the file within the loader
            archive = suffix
            #mode is the actual name we want to read
            name = mode
            if archive.is_package(name):  # use load_package with archive set
                m = self.load_package(fqname, pathname,
                                      archive_name=archive.archive)
                return m
            else:
                try:
                    co = archive.get_code(name)
                except ImportError:
                    cloudLog.warn("Cloud cannot read '%s' within '%s'. \
                        Import errors may result; \
                    please see PiCloud documentation." % (fqname,
                                                          archive.archive))
                    raise
                m = self.add_module(fqname, archive.archive, is_archive=True)
        else:
            if type == imp.PKG_DIRECTORY:
                m = self.load_package(fqname, pathname)
                self.msgout(2, "load_module ->", m)
                return m
            elif type == imp.PY_SOURCE:
                try:
                    co = compile(fp.read() + '\n', pathname, 'exec')
                except SyntaxError:  # compilation fail.
                    cloudLog.warn("Syntax error in %s. Import errors may\
                        occur in rare situations." % pathname)
                    raise ImportError("Syntax error in %s" % pathname)

            elif type == imp.PY_COMPILED:
                if fp.read(4) != imp.get_magic():
                    cloudLog.warn("Magic number on %s is invalid.  Import\
                        errors may occur in rare situations." % pathname)
                    self.msgout(2, "raise ImportError: Bad magic number",
                                pathname)
                    raise ImportError("Bad magic number in %s" % pathname)
                fp.read(4)
                co = marshal.load(fp)
            elif type == imp.C_EXTENSION:
                if fqname not in self.transitError:
                    cloudLog.warn("Cloud cannot transmit python extension\
                        '%s' located at '%s'.  Import errors may result;\
                        please see PiCloud documentation." % (fqname,
                                                              pathname))
                    self.transitError.add(fqname)
                raise ImportError(fqname)
            else:
                co = None
            m = self.add_module(fqname, filename=pathname)
        if co:
            if self.replace_paths:
                co = self.replace_paths_in_code(co)
            m.__code__ = co
            names = co.co_names
            if names and '__import__' in names:
                #PiCloud: Warn on __import__
                    cloudLog.warn('__import__ found within %s. Cloud cannot\
                    follow these \
                    dependencies. You MAY see importerror cloud exceptions.\
                    For more information,\
                    consult the PiCloud manual'
                    % fqname)
            self.scan_code(co, m)
        self.msgout(2, "load_module ->", m)
        return m

    def getUpdatedSnapshot(self):
        """Return any new myMods values since this was last called"""

        outList = []
        for modname, modobj in self.modules.items():
            if modname not in self.lastSnapshot:
                if modobj.is_archive:  # check if archive has already been sent
                    archive = modobj.__file__
                    if archive in self.lastSnapshot:
                        continue
                    else:
                        self.lastSnapshot.add(archive)
                outList.append((modobj.__file__, modobj.timestamp,
                                modobj.is_archive))
                self.lastSnapshot.add(modname)
        return outList


class FilePackager(object):
    """This class is responsible for the packaging of files"""
    """This is not thread safe"""

    fileCollection = None
    ARCHIVE_PATH = 'cloud.archive/'  # location where archives are extracted

    def __init__(self, path_infos=None):
        """path_infos is a list of (paths relative to site-packages,
        archive)"""
        self.fileCollection = {}
        if path_infos:
            for relPath, archive in path_infos:
                if archive:
                    self.addArchive(relPath)
                else:
                    self.addRelativePath(relPath)

    def addArchive(self, archive_name):
        for site in sys.path:
            if site.endswith(archive_name):
                self.fileCollection[self.ARCHIVE_PATH + archive_name] = site

    def addRelativePath(self, relPath):
        """Add a file by relative path to the File Transfer"""
        for site in sys.path:
            if site != '':
                site += '/'
            tst = os.path.join(site, relPath.encode())
            if os.path.exists(tst):
                self.fileCollection[relPath] = tst
                return
        from ..cloud import CloudException
        raise CloudException('FilePackager: %s not found on sys.path' %
            relPath)

    def getTarball(self):
        try:
            from cStringIO import StringIO
        except ImportError:
            from StringIO import StringIO
        import tarfile

        outfile = StringIO()
        tfile = tarfile.open(name='transfer.tar', fileobj=outfile, mode='w')
        tfile.dereference = True

        for arcname, fname in self.fileCollection.items():
            tfile.add(name=fname, arcname=arcname, recursive=False)
        tfile.close()

        return outfile.getvalue()
