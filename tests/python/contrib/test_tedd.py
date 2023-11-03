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
"""Configure pytest of Tensor Expression Debug Display"""
import re
import tvm
from tvm import te
from tvm import topi
from tvm import relay
from tvm.relay import testing
from tvm.relay.backend import Runtime, Executor


def findany(pattern, _str):
    matches = re.findall(pattern, _str)
    assert len(matches) > 0, "Pattern not found.\nPattern: " + pattern + "\nString:  " + _str


def checkdependency():
    # pylint: disable=import-outside-toplevel
    import pkg_resources

    # pylint: disable=E1133
    return not {"graphviz", "ipython"} - {pkg.key for pkg in pkg_resources.working_set}


def test_dfg():
    """Tests dataflow graph"""
    input_a = te.placeholder((1024, 4096), dtype="float32", name="input_a")
    input_b = topi.nn.softmax(input_a)
    # confirm lower works
    s = te.create_schedule([input_b.op])

    def verify():
        # pylint: disable=import-outside-toplevel
        from tvm.contrib import tedd

        _str = tedd.viz_dataflow_graph(s, False, "", True)
        # Check all edges are available
        findany(r"digraph \"Dataflow Graph\"", str)
        findany(r"Stage_0:O_0 -> Tensor_0_0", str)
        findany(r"Tensor_0_0 -> Stage_1:I_0", str)
        findany(r"Stage_1:O_0 -> Tensor_1_0", str)
        findany(r"Tensor_0_0 -> Stage_2:I_0", str)
        findany(r"Tensor_1_0 -> Stage_2:I_1", str)
        findany(r"Stage_2:O_0 -> Tensor_2_0", str)
        findany(r"Tensor_2_0 -> Stage_3:I_0", str)
        findany(r"Stage_3:O_0 -> Tensor_3_0", str)
        findany(r"Tensor_2_0 -> Stage_4:I_0", str)
        findany(r"Tensor_3_0 -> Stage_4:I_1", str)
        findany(r"Stage_4:O_0 -> Tensor_4_0", str)

    if checkdependency():
        verify()


def test_itervar_relationship_graph():
    """Tests itervars relationship graph"""
    n = te.var("n")
    m = te.var("m")
    input_a = te.placeholder((n, m), name="input_a")
    k = te.reduce_axis((0, m), "k")
    input_b = te.compute((n,), lambda i: te.sum(input_a[i, k], axis=k), name="input_b")

    s = te.create_schedule(input_b.op)
    s[input_b].split(input_b.op.reduce_axis[0], factor=16)

    def verify():
        # pylint: disable=import-outside-toplevel
        from tvm.contrib import tedd

        _str = tedd.viz_itervar_relationship_graph(s, False, "", True)
        findany(r"digraph \"IterVar Relationship Graph\"", str)
        findany(r"subgraph cluster_legend", str)
        # Check subgraphs for stages
        findany(r"subgraph cluster_Stage_0", str)
        findany(r"subgraph cluster_Stage_1", str)
        # Check itervars and their types
        findany(r"\(kDataPar\)\<br/\>T.Range\(0, n\)", str)
        findany(r"\(kCommReduce\)\<br/\>T.Range\(0, m\)", str)
        # Check the split node
        findany(r"Split_Relation_1_0 +.+\>Split", str)
        # Check all edges to/from the split node
        findany(r"IterVar_1_1:itervar -> Split_Relation_1_0:Input", str)
        findany(r"Split_Relation_1_0:Outer -> IterVar_1_2:itervar", str)
        findany(r"Split_Relation_1_0:Inner -> IterVar_1_3:itervar", str)

    if checkdependency():
        verify()


def test_schedule_tree():
    """Tests schedule tree"""
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")
    n = te.var("n")
    m = te.var("m")
    op_l = te.var("op_l")
    input_a = te.placeholder((n, m, op_l), name="input_a")
    result_b = te.compute((n, m, op_l), lambda bi, bj, bk: input_a[bi, bj, bk] + 1, name="result_b")
    op_r = te.reduce_axis((0, m), "op_r")
    result_c = te.compute(
        (
            n,
            m,
        ),
        lambda ci, cj: te.sum(result_b[ci, cj, op_r], axis=op_r),
        name="result_c",
    )
    s = te.create_schedule(result_c.op)
    s.cache_read(input_a, "shared", [result_b])
    s[result_b].vectorize(result_b.op.axis[-1])
    s[result_c].reorder(result_c.op.reduce_axis[0], result_c.op.axis[0])
    _, op_ki = s[result_c].split(result_c.op.reduce_axis[0], factor=16)
    result_c2 = s.rfactor(result_c, op_ki)
    s[result_c2].compute_at(s[result_c], s[result_c].op.axis[-1])
    s[result_c].bind(s[result_c].op.axis[0], block_x)
    s[result_c].bind(s[result_c].op.axis[1], thread_x)

    def verify():
        # pylint: disable=import-outside-toplevel
        from tvm.contrib import tedd

        _str = tedd.viz_schedule_tree(s, False, "", True)
        findany(r"digraph \"Schedule Tree\"", str)
        findany(r"subgraph cluster_legend", str)
        # Check the input_a_shared stage, including memory scope, itervars,
        # and compute
        findany(
            r"Stage_1.*A\.shared<br/>Scope: shared.+>0.+>"
            r"ax0.*\(kDataPar\).+>1.+ax1.*\(kDataPar\).+>2.+>ax2.*\(kDataPar\).+>"
            r"\[A[\[\(]ax0, ax1, ax2[\)\]]\]",
            str,
        )
        # Check itervars of types different from KDataPar
        findany(r"bk.*\(kVectorized\)", str)
        findany(r"r.outer.*\(kCommReduce\)", str)
        findany(r"label=ROOT", str)
        # Check the compute_at edge
        findany(r"Stage_1.*\[color\=\"\#000000\"\]", str)

    if checkdependency():
        verify()


@tvm.testing.requires_llvm
def test_tedd_with_schedule_record():
    """Test to build a nn model and check if all schedules could be generated"""

    def check_schedule(executor):
        # pylint: disable=import-outside-toplevel
        from tvm.contrib import tedd

        error = {}
        for func_name, func_meta in executor.function_metadata.items():
            # check converted op only
            if "main" not in func_name:
                primfunc = list(func_meta.relay_primfuncs.values())[0]
                schs = primfunc.attrs["schedule"].schedule_record
                for index in range(len(schs)):
                    try:
                        sch = schs[index].normalize()
                        tedd.viz_dataflow_graph(sch, False, "", True)
                        tedd.viz_itervar_relationship_graph(sch, False, "", True)
                        tedd.viz_schedule_tree(sch, False, "", True)
                    except:  # pylint: disable=W0702
                        if func_name not in error:
                            error[func_name] = []
                        error[func_name].append(index)

        assert not error, str(error)

    if checkdependency():
        relay_mod, params = testing.mobilenet.get_workload(batch_size=1, dtype="float32")
        target_llvm = tvm.target.Target("llvm")
        config = {"te.keep_schedule_record": True}

        with tvm.transform.PassContext(opt_level=3, config=config):
            aot_executor_factory = relay.build(
                relay_mod,
                target_llvm,
                runtime=Runtime("cpp"),
                executor=Executor("aot"),
                params=params,
            )
            graph_executor_factory = relay.build(
                relay_mod,
                target_llvm,
                params=params,
            )

        check_schedule(aot_executor_factory)
        check_schedule(graph_executor_factory)


if __name__ == "__main__":
    test_dfg()
    test_itervar_relationship_graph()
    test_schedule_tree()
    test_tedd_with_schedule_record()
