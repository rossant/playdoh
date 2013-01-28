"""
XRange serialization routines
CTypes Extraction code based on a stack overflow answer by Denis Otkidach

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

import ctypes


PyObject_HEAD = [('ob_refcnt', ctypes.c_size_t),
                 ('ob_type', ctypes.c_void_p)]


class XRangeType(ctypes.Structure):
    _fields_ = PyObject_HEAD + [
        ('start', ctypes.c_long),
        ('step', ctypes.c_long),
        ('len', ctypes.c_long),
    ]


def xrangeToCType(xrangeobj):
    """Cast a xrange to a C representation of it
    Fields modified in the C Representation effect the xrangeobj
    """
    return ctypes.cast(ctypes.c_void_p(id(xrangeobj)),
                                         ctypes.POINTER(XRangeType)).contents


"""Encoding (generally for placing within a JSON object):"""


def encodeMaybeXrange(obj, allowLists=True):
    """Encode an object that might be an xrange
    Supports lists (1 level deep) as well"""
    if isinstance(obj, xrange):
        c_xrange = xrangeToCType(obj)
        return ['xrange', c_xrange.start, c_xrange.step, c_xrange.len]
    if allowLists and isinstance(obj, list):
        return [encodeMaybeXrange(elm, allowLists=False) for elm in obj]
    return obj


def decodeMaybeXrange(obj, allowLists=True):
    """Decode a JSON-encoded object that might be an xrange"""
    if isinstance(obj, list):
        if len(obj) == 4 and obj[0] == 'xrange':  # an xrange object
            outrange = xrange(0)
            c_xrange = xrangeToCType(outrange)  # get pointer
            c_xrange.start = obj[1]
            c_xrange.step = obj[2]
            c_xrange.len = obj[3]
            return outrange
        elif allowLists:  # decode internal xranges
            return [decodeMaybeXrange(elm, allowLists=False) for elm in obj]
    return obj


def decodeXrangeList(obj):
    """Decode a list of elements and encoded xranges into a single list"""
    # outer layer might be an xrange
    obj = decodeMaybeXrange(obj, allowLists=False)
    if isinstance(obj, xrange):
        return list(obj)  # convert to list
    outlst = []
    for elem in obj:
        if isinstance(elem, list) and len(elem) == 4 and elem[0] == 'xrange':
            outlst.extend(decodeMaybeXrange(elem, allowLists=False))
        else:
            outlst.append(elem)
    return outlst


"""Support for piece-wise xranges - effectively a list of xranges"""


class piecewiseXrange(list):
    def __init__(self):
        self._xrangeMode = False

    def toXrangeMode(self):
        """Turn on xrange iterator"""
        self._xrangeMode = True

    def myIter(self):
        """Ideally, this would overload __iter__, but that presents
        massive pickling issues"""
        if self._xrangeMode:
            return self.xrangeIter
        else:
            return self.__iter__

    def xrangeIter(self):  # use a generator to iterate
        for elm in self.__iter__():
            if isinstance(elm, xrange):
                for subelm in elm:
                    yield subelm
            else:
                yield elm


def xrangeIter(lst):
    """Return correct iterator"""
    if isinstance(lst, piecewiseXrange):
        return lst.myIter()()
    else:
        return lst.__iter__()


def filterXrangeList(func, xrange_list):
    """ Input is a list of xranges and integers
        This returns a similarly formatted output where func(elem)
        evaluates to True
        If an xrange reduces to one element, it just inserts an integer
    """

    if not hasattr(xrange_list, '__iter__') or \
            isinstance(xrange_list, xrange):
        xrange_list = [xrange_list]

    single_range = 2  # if > 0, then outList is just a single xrange
    no_xrange_output = True  # outList has no xranges inserted

    outList = piecewiseXrange()
    for elm in xrange_list:
        if isinstance(elm, (int, long)):  # elm is
            if func(elm):
                outList.append(elm)
                single_range = 0  # individual elements present -
                                  # so not single xrange
        elif isinstance(elm, xrange):
            cxrange = xrangeToCType(elm)
            step = cxrange.step
            basenum = None
            for num in elm:  # iterate through xrange
                if func(num):
                    if basenum == None:
                        basenum = num
                else:
                    if basenum != None:  # push back xrange
                        # only one element: push an integer
                        if num - step == basenum:
                            outList.append(basenum)
                            single_range = 0
                        else:
                            outList.append(xrange(basenum, num, step))
                            single_range -= 1
                            no_xrange_output = False
                        basenum = None
            if basenum != None:  # cleanup
                num += step
                if num - step == basenum:  # only one element: push an integer
                    outList.append(basenum)
                    single_range = 0
                else:
                    outList.append(xrange(basenum, num, step))
                    single_range -= 1
                    no_xrange_output = False
        else:
            raise TypeError('%s (type %s) is not of type int, \
                long or xrange' % (elm, type(elm)))

    if outList:
        if not no_xrange_output:
            if single_range > 0:  # only one xrange appended - just return it
                return outList[0]
            else:
                outList.toXrangeMode()
    return outList


def iterateXRangeLimit(obj, limit):
    """Generate xrange lists based on some limit.
    Assumes obj is an xrange, or xrange list, or None
    May also return single xrange objects"""
    def innerGenerator(obj):
        if not obj:  # empty list/non case
            yield obj
            return

        if isinstance(obj, xrange):
            if len(obj) <= limit:
                yield obj
                return
            # use default algorithm if longer
            obj = [obj]

        # main algorithm
        outlist = []
        cnt = 0  # number of items appended
        for xr in obj:
            if isinstance(xr, (int, long)):
                outlist.append(xr)
                cnt += 1
                if cnt >= limit:
                    yield outlist
                    outlist = []
                    cnt = 0
            elif isinstance(xr, xrange):
                while True:
                    if len(xr) + cnt <= limit:
                        outlist.append(xr)
                        cnt += len(xr)
                        break
                    else:  # break apart xrange
                        oldlen = len(xr)
                        allowed = limit - cnt
                        c_xrange = xrangeToCType(xr)
                        breakpoint = c_xrange.start + c_xrange.step * allowed
                        to_app = xrange(c_xrange.start, breakpoint,\
                                        c_xrange.step)
                        outlist.append(to_app)
                        yield outlist
                        outlist = []
                        cnt = 0
                        xr = xrange(breakpoint, breakpoint + \
                        (oldlen - allowed) * c_xrange.step, c_xrange.step)
                if len(outlist) >= limit:
                    yield outlist
                    outlist = []

            else:
                raise TypeError('%s (type %s) is not of type int,\
                    long or xrange' % (xr, type(xr)))
        if outlist:
            yield outlist
    return innerGenerator(obj)

"""
Quick unit test
(TODO: Move to unit tests)
"""

if __name__ == '__main__':

    for m in [xrange(0, 10, 2), xrange(0, 1), xrange(0), xrange(0, -2),
                                    xrange(0, 6), xrange(0, 7, 3)]:
        me = encodeMaybeXrange(m)
        n = decodeMaybeXrange(me)
        print m, me, n

        #split
        print list(iterateXRangeLimit(m, 1000))
        print list(iterateXRangeLimit(m, 3))
        print [list(x[0]) if type(x) is list else x \
            for x in iterateXRangeLimit(m, 3)]

    xrl = filterXrangeList(lambda x: x % 10, xrange(10, 100))
    print xrl

    lmt = list(iterateXRangeLimit(xrl, 15))
    print lmt

    lmt_read = [[list(x) for x in y] for y in lmt]
    lmt2 = [reduce(lambda x, y: x + y, x) for x in lmt_read]
    print lmt2

    lenz = [len(x) for x in lmt2]
    print lenz

    print list(iterateXRangeLimit(range(25), 13))
