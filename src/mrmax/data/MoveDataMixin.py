"""MoveDataMixin."""

import dataclasses
from collections.abc import Callable, Iterator
from copy import copy as shallowcopy
from typing import Any, ClassVar, Protocol, Self, TypeAlias, TypeVar, cast, overload, runtime_checkable

import jax
import jax.numpy as jnp


class InconsistentDeviceError(ValueError):
    """Raised if the devices of different fields differ.

    There is no single device that all fields are on, thus
    the overall device of the object cannot be determined.
    """

    def __init__(self, *devices):
        """Initialize.

        Parameters
        ----------
        devices
            The devices of the fields that differ.
        """
        super().__init__(f'Inconsistent devices found, found at least {", ".join(str(d) for d in devices)}')


@runtime_checkable
class DataclassInstance(Protocol):
    """An instance of a dataclass."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


T = TypeVar('T')

# Type aliases for common types
ParsedType: TypeAlias = tuple[str | jax.Device | None, jnp.dtype | None, bool, bool]


class MoveDataMixin:
    """Move dataclass fields to cpu/gpu and convert dtypes."""

    @overload
    def to(
        self,
        device: str | jax.Device | None = None,
        dtype: jnp.dtype | None = None,
        *,
        copy: bool = False,
    ) -> Self: ...

    @overload
    def to(
        self,
        dtype: jnp.dtype,
        *,
        copy: bool = False,
    ) -> Self: ...

    @overload
    def to(
        self,
        array: jnp.ndarray,
        *,
        copy: bool = False,
    ) -> Self: ...

    def to(self, *args, **kwargs) -> Self:
        """Perform dtype and/or device conversion of data.

        A `jnp.dtype` and `jax.Device` are inferred from the arguments
        args and kwargs. Please have a look at the
        documentation of `jnp.ndarray` for more details.

        A new instance of the dataclass will be returned.

        The conversion will be applied to all JAX array fields of the dataclass,
        and to all fields that implement the `MoveDataMixin`.

        The dtype-type, i.e. float or complex will always be preserved,
        but the precision of floating point dtypes might be changed.

        Example:
        If called with ``dtype=jnp.float32`` OR ``dtype=jnp.complex64``:

        - A ``complex128`` array will be converted to ``complex64``
        - A ``float64`` array will be converted to ``float32``
        - A ``bool`` array will remain ``bool``
        - An ``int64`` array will remain ``int64``

        If other conversions are desired, please use the `~jnp.ndarray.astype` method of
        the fields directly.

        If the copy argument is set to `True` (default), a deep copy will be returned
        even if no conversion is necessary.
        If two fields are views of the same data before, in the result they will be independent
        copies if copy is set to `True` or a conversion is necessary.
        If set to `False`, some arrays might be shared between the original and the new object.
        """

        # Parse the arguments of the three overloads and call _to with the parsed arguments
        def parse3(
            other: jnp.ndarray,
            copy: bool = False,
        ) -> ParsedType:
            return other.device, other.dtype, copy, False

        def parse2(
            dtype: jnp.dtype,
            copy: bool = False,
        ) -> ParsedType:
            return None, dtype, copy, False

        def parse1(
            device: str | jax.Device | None = None,
            dtype: jnp.dtype | None = None,
            copy: bool = False,
        ) -> ParsedType:
            return device, dtype, copy, False

        if (args and isinstance(args[0], jnp.ndarray)) or 'array' in kwargs:
            # overload 3 ("array" specifies the dtype and device)
            device, dtype, copy, shared_memory = parse3(*args, **kwargs)
        elif args and isinstance(args[0], jnp.dtype):
            # overload 2 (no device specified, only dtype)
            device, dtype, copy, shared_memory = parse2(*args, **kwargs)
        else:
            # overload 1 (device and dtype specified)
            device, dtype, copy, shared_memory = parse1(*args, **kwargs)
        return self._to(device=device, dtype=dtype, copy=copy, shared_memory=shared_memory)

    def _items(self) -> Iterator[tuple[str, Any]]:
        """Return an iterator over fields of the object."""
        if isinstance(self, DataclassInstance):
            for field in dataclasses.fields(self):
                name = field.name
                data = getattr(self, name)
                yield name, data

    def _to(
        self,
        device: jax.Device | str | None = None,
        dtype: jnp.dtype | None = None,
        shared_memory: bool = False,
        copy: bool = False,
        memo: dict | None = None,
    ) -> Self:
        """Move data to device and convert dtype if necessary.

        This method is called by `.to()`, `.cuda()`, `.cpu()`,
        `.double()`, and so on. It should not be called directly.

        See `MoveDataMixin.to()` for more details.

        Parameters
        ----------
        device
            The destination device.
        dtype
            The destination dtype.
        shared_memory
            If `True` and the target device is CPU, the arrays will reside in shared memory.
            Otherwise, the argument has no effect.
        copy
            If `True`, the returned array will always be a copy, even if the input was already on the correct device.
            This will also create new arrays for views.
        memo
            A dictionary to keep track of already converted objects to avoid multiple conversions.
        """
        new = shallowcopy(self) if copy else self

        if memo is None:
            memo = {}

        def _array_to(data: jnp.ndarray) -> jnp.ndarray:
            """Move array to device and convert dtype if necessary."""
            new_dtype: jnp.dtype | None
            if (dtype is not None and jnp.issubdtype(data.dtype, jnp.floating)) or (
                dtype is not None and jnp.issubdtype(data.dtype, jnp.complexfloating)
            ):
                new_dtype = dtype
            else:
                # bool or int: keep as is
                new_dtype = None
            data = jax.device_put(data, device) if device is not None else data
            data = data.astype(new_dtype) if new_dtype is not None else data
            if shared_memory and device is not None and isinstance(device, str) and device == 'cpu':
                data = jax.device_put(data, jax.devices('cpu')[0])
            return data

        def _mixin_to(obj: MoveDataMixin) -> MoveDataMixin:
            return obj._to(
                device=device,
                dtype=dtype,
                shared_memory=shared_memory,
                copy=copy,
                memo=memo,
            )

        def _convert(data: T) -> T:
            converted: Any  # https://github.com/python/mypy/issues/10817
            if isinstance(data, jnp.ndarray):
                converted = _array_to(data)
            elif isinstance(data, MoveDataMixin):
                converted = _mixin_to(data)
            else:
                converted = data
            return cast(T, converted)

        for name, data in self._items():
            if id(data) in memo:
                setattr(new, name, memo[id(data)])
            else:
                converted_data = _convert(data)
                setattr(new, name, converted_data)
                memo[id(data)] = converted_data

        return new

    def apply(
        self: Self,
        function: Callable[[Any], Any] | None = None,
        *,
        recurse: bool = True,
    ) -> Self:
        """Apply a function to all children. Returns a new object.

        Parameters
        ----------
        function
            The function to apply to all fields. `None` is interpreted as a no-op.
        recurse
            If `True`, the function will be applied to all children that are `MoveDataMixin` instances.
        """
        new = self.clone().apply_(function, recurse=recurse)
        return new

    def apply_(
        self: Self,
        function: Callable[[Any], Any] | None = None,
        *,
        memo: dict[int, Any] | None = None,
        recurse: bool = True,
    ) -> Self:
        """Apply a function to all children in-place.

        Parameters
        ----------
        function
            The function to apply to all fields. `None` is interpreted as a no-op.
        memo
            A dictionary to keep track of objects that the function has already been applied to,
            to avoid multiple applications. This is useful if the object has a circular reference.
        recurse
            If `True`, the function will be applied to all children that are `MoveDataMixin` instances.
        """
        applied: Any

        if memo is None:
            memo = {}

        if function is None:
            return self

        for name, data in self._items():
            if id(data) in memo:
                # this works even if self is frozen
                object.__setattr__(self, name, memo[id(data)])
                continue
            if recurse and isinstance(data, MoveDataMixin):
                applied = data.apply_(function, memo=memo)
            else:
                applied = function(data)
            memo[id(data)] = applied
            object.__setattr__(self, name, applied)
        return self

    def cuda(
        self,
        device: jax.Device | str | None = None,
        *,
        copy: bool = False,
    ) -> Self:
        """Put object in CUDA memory.

        Parameters
        ----------
        device
            The destination GPU device. Defaults to the current CUDA device.
        copy
            If `True`, the returned array will always be a copy, even if the input was already on the correct device.
            This will also create new arrays for views.
        """
        if device is None:
            device = jax.devices('cuda')[0]
        return self._to(device=device, dtype=None, copy=copy)

    def cpu(self, *, copy: bool = False) -> Self:
        """Put in CPU memory.

        Parameters
        ----------
        copy
            If `True`, the returned array will always be a copy, even if the input was already on the correct device.
            This will also create new arrays for views.
        """
        return self._to(device='cpu', dtype=None, copy=copy)

    def double(self, *, copy: bool = False) -> Self:
        """Convert all float arrays to double precision.

        converts ``float`` to ``float64`` and ``complex`` to ``complex128``

        Parameters
        ----------
        copy
            If `True`, the returned array will always be a copy, even if the input was already on the correct device.
            This will also create new arrays for views.
        """
        return self._to(dtype=jnp.float64, copy=copy)

    def half(self, *, copy: bool = False) -> Self:
        """Convert all float arrays to half precision.

        converts ``float`` to ``float16`` and ``complex`` to ``complex32``

        Parameters
        ----------
        copy
            If `True`, the returned array will always be a copy, even if the input was already on the correct device.
            This will also create new arrays for views.
        """
        return self._to(dtype=jnp.float16, copy=copy)

    def single(self, *, copy: bool = False) -> Self:
        """Convert all float arrays to single precision.

        converts ``float`` to ``float32`` and ``complex`` to ``complex64``

        Parameters
        ----------
        copy
            If `True`, the returned array will always be a copy, even if the input was already on the correct device.
            This will also create new arrays for views.
        """
        return self._to(dtype=jnp.float32, copy=copy)

    @property
    def device(self) -> jax.Device | None:
        """Return the device of the arrays.

        Looks at each field of a dataclass implementing a device attribute,
        such as `jnp.ndarray` or `MoveDataMixin` instances. If the devices
        of the fields differ, an :py:exc:`~mrmax.data.InconsistentDeviceError` is raised, otherwise
        the device is returned. If no field implements a device attribute,
        None is returned.

        Raises
        ------
        :py:exc:`InconsistentDeviceError`
            If the devices of different fields differ.

        Returns
        -------
            The device of the fields or `None` if no field implements a `device` attribute.
        """
        device: None | jax.Device = None
        for _, data in self._items():
            if not hasattr(data, 'device'):
                continue
            current_device = getattr(data, 'device', None)
            if current_device is None:
                continue
            if device is None:
                device = current_device
            elif device != current_device:
                raise InconsistentDeviceError(current_device, device)
        return device

    def clone(self: Self) -> Self:
        """Return a deep copy of the object."""
        return self._to(device=None, dtype=None, copy=True)

    @property
    def is_cuda(self) -> bool:
        """Return `True` if all arrays are on a single CUDA device.

        Checks all array attributes of the dataclass for their device,
        (recursively if an attribute is a `MoveDataMixin`)

        Returns `False` if not all arrays are on the same CUDA devices, or if the device is inconsistent,
        returns `True` if the data class has no arrays as attributes.
        """
        try:
            device = self.device
        except InconsistentDeviceError:
            return False
        if device is None:
            return True
        return device.platform == 'cuda'

    @property
    def is_cpu(self) -> bool:
        """Return True if all arrays are on the CPU.

        Checks all array attributes of the dataclass for their device,
        (recursively if an attribute is a `MoveDataMixin`)

        Returns `False` if not all arrays are on cpu or if the device is inconsistent,
        returns `True` if the data class has no arrays as attributes.
        """
        try:
            device = self.device
        except InconsistentDeviceError:
            return False
        if device is None:
            return True
        return device.platform == 'cpu'
