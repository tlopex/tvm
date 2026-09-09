# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Constraint snapshots used by ThreadSync; these tests do not require a GPU."""

import pytest

import tvm
import tvm.testing
from tvm import s_tir, tirx
from tvm.script import tirx as T


def apply_sync(func):
    return s_tir.transform.ThreadSync("shared")(tvm.IRModule({"main": func}))["main"]


def sync_count(func):
    calls = []

    def visit(node):
        if isinstance(node, tvm.ir.Call) and node.op == tvm.ir.Op.get("tirx.tvm_storage_sync"):
            calls.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return len(calls)


def test_disjoint_even_odd_indices():
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((64,), "float32"),
        B: T.Buffer((64,), "float32"),
        Out: T.Buffer((64,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((128,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        S[2 * tx + 1] = A[tx]
        S[2 * tx] = B[tx]
        Out[tx] = S[2 * tx + 1]

    tvm.ir.assert_structural_equal(apply_sync(func), func)


def test_disjoint_branch_constraints():
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((64,), "float32"),
        B: T.Buffer((64,), "float32"),
        Out: T.Buffer((32,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((64,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        S[tx] = A[tx]
        T.tvm_storage_sync("shared")
        if tx < 32:
            S[tx] = B[tx]
        if tx < 32:
            Out[tx] = S[tx + 32]

    tvm.ir.assert_structural_equal(apply_sync(func), func)


def test_cross_thread_bind_requires_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((64,), "float32"), Out: T.Buffer((64,), "float32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((64,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        j = T.bind(tx)
        S[j] = A[tx]
        Out[tx] = S[(j + 1) % 64]

    # Equating j in two executions would falsely prove the indices unequal.
    assert sync_count(apply_sync(func)) == 1


def test_opposite_branches_can_conflict_across_threads():
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((64,), "float32"),
        B: T.Buffer((64,), "float32"),
        Out: T.Buffer((64,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((64,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        S[tx] = A[tx]
        T.tvm_storage_sync("shared")
        if tx < 32:
            S[tx] = B[tx]
        if tx >= 32:
            Out[tx] = S[tx - 32]

    assert sync_count(apply_sync(func)) == 2


@pytest.mark.parametrize("lazy", [False, True])
def test_select_and_if_then_else(lazy):
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((64,), "float32"),
        B: T.Buffer((64,), "float32"),
        Out: T.Buffer((64,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((128,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        S[tx] = A[tx]
        S[tx + 64] = A[tx]
        T.tvm_storage_sync("shared")
        if tx < 32:
            S[tx] = B[tx]
        Out[tx] = T.Select(tx < 32, S[tx + 64], S[tx + 1])

    if lazy:

        def replace(node):
            if isinstance(node, tirx.Select):
                return tirx.if_then_else(node.condition, node.true_value, node.false_value)
            return None

        func = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, replace))

    # An eager Select still reads S[tx + 1] in the lower half of the block.
    assert sync_count(apply_sync(func)) == (1 if lazy else 2)


def test_mutable_bind_values_are_not_equated():
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((64,), "float32"),
        Index: T.Buffer((64,), "int32"),
        Out: T.Buffer((64,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((128,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        a = T.bind(Index[tx])
        Index[tx] = a + 1
        b = T.bind(Index[tx])
        S[2 * tx + 1] = A[tx]
        S[2 * tx] = A[tx]
        Out[tx] = S[2 * ((tx + 1) % 64) + (a - b + 1) % 2]

    # Treating a and b as the same read would turn the read index into an odd
    # number.  In this program b == a + 1 and the read conflicts across threads.
    assert sync_count(apply_sync(func)) == 1


def test_branch_constraints_do_not_escape():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((64,), "float32"), Out: T.Buffer((64,), "float32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((64,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        if tx < 32:
            T.evaluate(0)
        S[tx] = A[tx]
        Out[tx] = S[(tx + 32) % 64]

    assert sync_count(apply_sync(func)) == 1


def test_bind_survives_nested_sequence():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((64,), "float32"), Out: T.Buffer((64,), "float32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((128,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        j = T.bind(2 * tx)
        S[2 * tx + 1] = A[tx]
        S[j] = A[tx]
        Out[tx] = S[2 * tx + 1]

    def nest_bind(node):
        if isinstance(node, tirx.Bind):
            return tirx.SeqStmt([node, tirx.Evaluate(0)])
        return None

    func = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, nest_bind))
    tvm.ir.assert_structural_equal(apply_sync(func), func)


def test_condition_reads_are_recorded():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((64,), "int32"), Out: T.Buffer((64,), "int32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((64,), "int32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        S[tx] = A[tx]
        if S[(tx + 1) % 64] > 0:
            Out[tx] = 1

    assert sync_count(apply_sync(func)) == 1


def test_same_thread_keeps_no_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((64,), "float32"), Out: T.Buffer((64,), "float32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((64,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        S[tx] = A[tx]
        Out[tx] = S[tx]

    tvm.ir.assert_structural_equal(apply_sync(func), func)


def test_loop_summary_keeps_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((128,), "float32"), Out: T.Buffer((64,), "float32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((128,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        for i in range(2):
            S[2 * tx + i] = A[2 * tx + i]
        Out[tx] = S[2 * ((tx + 1) % 64)]

    assert sync_count(apply_sync(func)) == 1


def test_vector_access_keeps_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((128,), "float32"), Out: T.Buffer((64,), "float32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((128,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        S[T.Ramp(2 * tx, 1, 2)] = A[T.Ramp(2 * tx, 1, 2)]
        Out[tx] = S[2 * ((tx + 1) % 64) + 1]

    assert sync_count(apply_sync(func)) == 1


def test_offset_alias_keeps_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((128,), "float32"), Out: T.Buffer((64,), "float32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((128,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        Shifted = T.decl_buffer((127,), "float32", data=S.data, elem_offset=1, scope="shared")
        S[2 * tx] = A[tx]
        S[2 * tx + 1] = A[tx + 64]
        Out[tx] = Shifted[2 * ((tx + 1) % 64)]

    # Logical even indices in Shifted alias odd indices in S.
    assert sync_count(apply_sync(func)) == 1


def assert_sync_before_last_thread_statement(func):
    """Check both the number and placement of the required inter-thread barrier."""
    result = apply_sync(func)
    thread_bodies = []

    def visit(node):
        if (
            isinstance(node, tirx.AttrStmt)
            and node.attr_key == "thread_extent"
            and node.node.thread_tag == "threadIdx.x"
        ):
            thread_bodies.append(node.body)

    tirx.stmt_functor.post_order_visit(result.body, visit)
    assert len(thread_bodies) == 1

    def flatten(stmt):
        if isinstance(stmt, tirx.SeqStmt):
            return [child for part in stmt.seq for child in flatten(part)]
        return [stmt]

    statements = flatten(thread_bodies[0])
    barrier = statements[-2]
    assert isinstance(barrier, tirx.Evaluate)
    assert isinstance(barrier.value, tvm.ir.Call)
    assert barrier.value.op == tvm.ir.Op.get("tirx.tvm_storage_sync")
    assert sync_count(result) == 2


def test_narrowing_cast_index_keeps_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((256,), "float32"),
        B: T.Buffer((256,), "float32"),
        Out: T.Buffer((256,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((256,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 256)
        S[tx] = A[tx]
        T.tvm_storage_sync("shared")
        if tx < 128:
            S[tx] = B[tx]
        if tx >= 128:
            Out[tx] = S[T.Cast("int32", T.Cast("uint8", tx + 128))]

    # Reader 128 reads S[0], which writer 0 updates. A narrowing cast maps
    # [256, 383] to [0, 127]; intersecting with [0, 255] is not a valid bound.
    assert_sync_before_last_thread_statement(func)


def test_narrowing_cast_bind_keeps_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((256,), "float32"),
        B: T.Buffer((256,), "float32"),
        Out: T.Buffer((256,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((256,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 256)
        S[tx] = A[tx]
        T.tvm_storage_sync("shared")
        j = T.bind(T.Cast("int32", T.Cast("uint8", tx + 128)))
        k = T.bind(j)
        if tx < 128:
            S[tx] = B[tx]
        if tx >= 128:
            Out[tx] = S[k]

    # Checking only the index expression (k) misses the cast in its definition.
    assert_sync_before_last_thread_statement(func)


def test_narrowing_cast_condition_keeps_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((256,), "float32"),
        B: T.Buffer((256,), "float32"),
        Out: T.Buffer((256,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((256,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 256)
        S[tx] = A[tx]
        T.tvm_storage_sync("shared")
        if tx < 128:
            S[tx] = B[tx]
        if T.Cast("int32", T.Cast("uint8", tx + 128)) < 128:
            Out[tx] = S[(tx + 128) % 256]

    # The condition is true for readers 128..255, which read slots 0..127.
    assert_sync_before_last_thread_statement(func)


def test_narrowing_cast_assert_keeps_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((256,), "float32"),
        B: T.Buffer((256,), "float32"),
        Out: T.Buffer((256,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((256,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 256)
        S[tx] = A[tx]
        T.tvm_storage_sync("shared")
        if tx < 128:
            S[tx] = B[tx]
        if tx >= 128:
            assert T.Cast("int32", T.Cast("uint8", tx + 128)) < 128, "wrapped index"
            Out[tx] = S[tx - 128]

    assert_sync_before_last_thread_statement(func)


def test_signed_narrowing_cast_keeps_sync():
    @T.prim_func(private=True, s_tir=True)
    def func(
        A: T.Buffer((64,), "float32"),
        B: T.Buffer((64,), "float32"),
        Out: T.Buffer((64,), "float32"),
    ):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((64,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        S[tx] = A[tx]
        T.tvm_storage_sync("shared")
        if tx < 32:
            S[tx] = B[tx]
        if tx >= 32:
            Out[tx] = S[T.Cast("int32", T.Cast("int64", tx) + T.int64(4294967264))]

    # Both types are signed index types, but int64 -> int32 still narrows.
    # Truncation makes reader 32 read the location written by thread 0.
    assert_sync_before_last_thread_statement(func)


def test_signed_widening_cast_preserves_disjoint_proof():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((64,), "float32"), Out: T.Buffer((64,), "float32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        S = T.alloc_buffer((128,), "float32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        j = T.bind(T.Cast("int64", tx) * T.int64(2))
        S[2 * tx + 1] = A[tx]
        S[j] = A[tx]
        Out[tx] = S[2 * tx + 1]

    tvm.ir.assert_structural_equal(apply_sync(func), func)


if __name__ == "__main__":
    tvm.testing.main()
