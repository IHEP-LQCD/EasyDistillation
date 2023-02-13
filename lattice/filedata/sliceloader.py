import io
from typing import Tuple, Union

import numpy


class binloader:
    filename: str
    dtype: numpy.dtype
    offset: int
    shape: tuple

    def __init__(
        self,
        filename: str,
        dtype: Union[str, type, numpy.dtype] = numpy.uint8,
        offset: int = 0,
        shape: Tuple[int] = None
    ):
        if not isinstance(dtype, numpy.dtype):
            descr = numpy.dtype(dtype)
        else:
            descr = dtype

        with open(filename, "rb") as fid:
            fid.seek(0, 2)
            flen = fid.tell()
            dbytes = descr.itemsize
            bytes = 0

            if shape is None:
                bytes = flen - offset
                if bytes % dbytes:
                    raise ValueError("Size of available data is not a multiple of the data-type size.")
                size = bytes // dbytes
                shape = (size, )
            elif not isinstance(shape, tuple):
                shape = tuple(shape, )
                size = numpy.intp(1)
                for k in shape:
                    size *= k
                bytes = offset + size * dbytes

            if flen < bytes:
                raise ValueError("Size of available data is less than acquired by dtype and shape.")

        self.filename = filename
        self.dtype = descr
        self.offset = offset
        self.shape = shape

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item, )

        shape = self.shape
        offset = self.offset
        ndim = len(shape)
        stride = [1]
        for s in shape[::-1][:-1]:
            stride.append(stride[-1] * s)
        stride = stride[::-1]

        resshape = []
        resitem = []
        resstride = []
        full = [False] * ndim
        continuum = [False] * ndim
        skip = 0
        i = -1
        for i, dim in enumerate(item):
            if isinstance(dim, int):
                skip += dim * stride[i]
            elif isinstance(dim, slice):
                if dim.start is None:
                    start = 0
                else:
                    start = dim.start % shape[i]
                if dim.stop is None:
                    stop = shape[i]
                elif dim.stop < 0:
                    stop = dim.stop % shape[i]
                else:
                    stop = min(dim.stop, shape[i])
                if dim.step is None:
                    step = 1
                else:
                    step = dim.step

                resshape.append((stop - start) // step)
                resitem.append(slice(0, stop - start, step))
                resstride.append(int(numpy.prod(shape[i + 1:])))
                skip += start * stride[i]
                if step == 1:
                    continuum[i] = True
                    if stop - start == shape[i]:
                        full[i] = True
            else:
                try:
                    dim_list = list(dim)
                    resshape.append(len(dim_list))
                    resitem.append(dim_list)
                    resstride.append(int(numpy.prod(shape[i + 1:])))
                except TypeError:
                    raise ValueError("Index in [] should be int, slice or iterable.")

        i += 1
        while i < ndim:
            resshape.append(shape[i])
            resitem.append(slice(0, shape[i], 1))
            resstride.append(int(numpy.prod(shape[i + 1:])))
            continuum[i] = True
            full[i] = True
            i += 1

        rescount = 1
        i = ndim - 1
        while i != 0 and full[i]:
            rescount *= shape[i]
            i -= 1
        if continuum[i]:
            rescount *= resshape[i - ndim]
        else:
            i += 1

        resoffset = offset + skip * self.dtype.itemsize
        resndim = len(resshape)
        realshape = resshape[:i - ndim + resndim]
        realitem = resitem[:i - ndim + resndim]
        realstride = resstride[:i - ndim + resndim]

        return self.load(resshape, realitem, realshape, realstride, rescount, resoffset)

    def load(self, resshape, item, shape, stride, count, offset):
        ndim = len(shape)
        index = [0] * ndim
        dbytes = self.dtype.itemsize

        res = numpy.zeros([int(numpy.prod(shape)), count], self.dtype)

        with open(self.filename, "rb") as fid:
            if ndim == 0:
                fid.seek(offset, io.SEEK_SET)
                res = numpy.fromfile(fid, self.dtype, count)
                return res.reshape(resshape)
            else:
                res = numpy.zeros([int(numpy.prod(shape)), count], self.dtype)
                index[-1] = -1
                resindex = 0
                while True:
                    i = ndim - 1
                    index[i] += 1
                    while index[i] == shape[i] and i > 0:
                        index[i] = 0
                        index[i - 1] += 1
                        i -= 1
                    if i == 0 and index[0] == shape[0]:
                        break
                    skip = 0
                    for i in range(ndim):
                        it = item[i]
                        if isinstance(it, slice):
                            skip += index[i] * it.step * stride[i]
                        else:
                            skip += it[index[i]] * stride[i]
                    fid.seek(offset + skip * dbytes, io.SEEK_SET)
                    res[resindex] = numpy.fromfile(fid, self.dtype, count)
                    resindex += 1
                return res.reshape(resshape)


class npyloader:
    def __init__(self, filename: str) -> None:
        magic_prefix = numpy.lib.format.MAGIC_PREFIX

        with open(filename, "rb") as fid:
            magic = fid.read(len(magic_prefix))
            if magic != magic_prefix:
                raise ValueError(f"{filename} is not a .npy file.")
            fid.seek(0, io.SEEK_SET)
            version = numpy.lib.format.read_magic(fid)
            if version not in [(1, 0), (2, 0), (3, 0), None]:
                raise ValueError(f"We only support format version (1,0), (2,0), and (3,0), not {version}.")
            shape, fortran_order, dtype = numpy.lib.format._read_array_header(fid, version)
            offset = fid.tell()

        self.loader = binloader(filename, dtype, offset, shape)
        self.dtype = dtype
        self.offset = offset
        self.shape = shape
        self.fortran_order = fortran_order
        #     if len(shape) == 0:
        #         count = 1
        #     else:
        #         count = numpy.multiply.reduce(shape, dtype=numpy.int64)

        # if fortran_order:
        #     array.shape = shape[::-1]
        #     array = array.transpose()
        # else:
        #     array.shape = shape

    def __getitem__(self, item):
        return self.loader[item]
