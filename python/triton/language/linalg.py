from .core import builtin, add, _unwrap_if_constexpr
from .semantic import *

@builtin
def elementwise_add(a, b, _builder=None):
  x = _unwrap_if_constexpr(a)
  y = _unwrap_if_constexpr(b)
  x, y = binary_op_type_checking_impl(x, y, _builder, False, False)
  x_scalar_ty = x.type.scalar
  return tl.tensor(_builder.create_linalg_elemwise_binary_add(x.handle, y.handle), x.type)
  # int + int
  #elif x_scalar_ty.is_int():
  #  return tl.tensor(builder.create_add(x.handle, y.handle), x.type)
  raise TypeError(f"unexpected type {x_scalar_ty}")
