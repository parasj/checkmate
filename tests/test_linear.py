from remat.core.dfgraph import gen_linear_graph


def test_checkpointall():
    for graph_length in [2, 4, 8, 16]:
        g = gen_linear_graph(graph_length)
        assert g.size_fwd == graph_length
