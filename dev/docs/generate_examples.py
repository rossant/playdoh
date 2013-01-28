import os, re, glob, inspect, compiler, unicodedata, fnmatch
os.chdir('../../examples')

class GlobDirectoryWalker:
    # a forward iterator that traverses a directory tree

    def __init__(self, directory, pattern="*"):
        self.stack = [directory]
        self.pattern = pattern
        self.files = []
        self.index = 0

    def __getitem__(self, index):
        while 1:
            try:
                file = self.files[self.index]
                self.index = self.index + 1
            except IndexError:
                # pop next directory from stack
                self.directory = self.stack.pop()
                self.files = os.listdir(self.directory)
                self.index = 0
            else:
                # got a filename
                fullname = os.path.join(self.directory, file)
                if os.path.isdir(fullname) and not os.path.islink(fullname):
                    self.stack.append(fullname)
                if fnmatch.fnmatch(file, self.pattern):
                    return fullname

examplesfnames = [fname for fname in GlobDirectoryWalker('.', '*.py') if 'external' not in fname]
examplespaths = []
examplesbasenames = []
for f in examplesfnames:
    path, file = os.path.split(f)
    path = os.path.normpath(path)
    if path == '.': path = ''
    else: path = path + '_'
    filebase, ext = os.path.splitext(file)
    examplespaths.append(path)
    examplesbasenames.append(filebase)
examplescode = [open(fname, 'r').read() for fname in examplesfnames]
examplesdocs = []
examplesafterdoccode = []
examplesdocumentablenames = []
for code in examplescode:
    codesplit = code.split('\n')
    readingdoc = False
    doc = []
    afterdoccode = ''
    for i in range(len(codesplit)):
        stripped = codesplit[i].strip()
        if stripped[:3] == '"""' or stripped[:3] == "'''":
            if not readingdoc:
                readingdoc = True
            else:
                afterdoccode = '\n'.join(codesplit[i + 1:])
                break
        elif readingdoc:
            doc.append(codesplit[i])
        elif not stripped or stripped[0] == '#':
            pass
        else:
            break
    doc = '\n'.join(doc)
    # next line replaces unicode characters like e-acute with standard ascii representation
    examplesdocs.append(unicodedata.normalize('NFKD', unicode(doc, 'latin-1')).encode('ascii', 'ignore'))
    examplesafterdoccode.append(afterdoccode)
    examplesdocumentablenames.append([])
#    try:
#        examplesdocumentablenames.append(set(compiler.compile(code, '', 'exec').co_names) & documentable_names)
#    except SyntaxError:
#        print code
#        raise
examples = zip(examplesfnames, examplespaths, examplesbasenames, examplescode, examplesdocs, examplesafterdoccode, examplesdocumentablenames)
os.chdir('../docs_sphinx')
for fname, path, basename, code, docs, afterdoccode, documentables in examples:
    title = 'Example: ' + basename
    if len(path): title += ' (' + path[:-1] + ')'
    output = '.. currentmodule:: playdoh\n\n'
    output += '.. _example-' + path + basename + ':\n\n'
    if len(documentables):
        output += '.. index::\n'
        for dname in documentables:
            output += '   pair: example usage; ' + dname + '\n'
        output += '\n'
    output += title + '\n' + '=' * len(title) + '\n\n'
    output += docs + '\n\n::\n\n'
    output += '\n'.join(['    ' + line for line in afterdoccode.split('\n')])
    output += '\n\n'
    open('examples-' + path + basename + '.txt', 'w').write(output)

