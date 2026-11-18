from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.graph.graph_utils import (
    build_minimal_graphdata_from_edge_list,
    sample_from_master_graphdata,
)


def _canonical_edges(dataframe):
    output = dataframe.copy()
    output["a"] = output[
        ["source", "target"]
    ].min(axis=1)
    output["b"] = output[
        ["source", "target"]
    ].max(axis=1)

    return (
        output[
            ["a", "b", "weight", "contact_type"]
        ]
        .sort_values(
            ["a", "b", "contact_type"]
        )
        .reset_index(drop=True)
    )


def test_graphdata_population_size(toy_model_factory):
    graphdata = toy_model_factory.graphdata

    assert graphdata.N == 6
    assert graphdata.adj_matrix.shape == (6, 6)
    assert len(graphdata.ages) == 6
    assert len(graphdata.degrees_arr) == 6


def test_neighbor_map_is_symmetric(toy_model_factory):
    graphdata = toy_model_factory.graphdata

    for source, neighbors in (
        graphdata.neighbor_map.items()
    ):
        for target, weight, contact_type in neighbors:
            reverse = graphdata.neighbor_map[target]

            assert any(
                reverse_target == source
                and reverse_weight == weight
                and reverse_type == contact_type
                for (
                    reverse_target,
                    reverse_weight,
                    reverse_type,
                ) in reverse
            )


def test_contact_type_csrs_match_edge_list(
    toy_model_factory,
):
    graphdata = toy_model_factory.graphdata

    for edge in graphdata.edge_list.itertuples(
        index=False
    ):
        indptr, indices, weights = (
            graphdata.csr_by_type[
                edge.contact_type
            ]
        )

        start = int(indptr[edge.source])
        end = int(indptr[edge.source + 1])

        source_targets = indices[start:end]
        source_weights = weights[start:end]

        matches = np.where(
            source_targets == edge.target
        )[0]

        assert matches.size == 1
        assert np.isclose(
            source_weights[matches[0]],
            edge.weight,
        )


def test_sampled_edges_are_valid_and_undirected_unique(
    toy_model_factory,
):
    config = toy_model_factory.config(
        population_overrides={
            "wp_contacts": 2,
            "sch_contacts": 2,
            "gq_contacts": 2,
            "cas_contacts": 2,
        },
        sim_overrides={
            "master_casual_candidates": 10,
        },
    )

    node_count = 6

    master_edges = pd.DataFrame(
        [
            (source, target, 1.0, "cas")
            for source in range(node_count)
            for target in range(
                source + 1,
                node_count,
            )
        ],
        columns=[
            "source",
            "target",
            "weight",
            "contact_type",
        ],
    )

    minimal = (
        build_minimal_graphdata_from_edge_list(
            master_edges,
            N=node_count,
        )
    )
    sampled = sample_from_master_graphdata(
        minimal,
        config,
        seed=123,
    )

    assert sampled["source"].between(
        0,
        node_count - 1,
    ).all()
    assert sampled["target"].between(
        0,
        node_count - 1,
    ).all()
    assert (
        sampled["source"]
        != sampled["target"]
    ).all()

    pairs = sampled.apply(
        lambda row: tuple(
            sorted(
                (
                    int(row["source"]),
                    int(row["target"]),
                )
            )
        ),
        axis=1,
    )

    assert not pairs.duplicated().any()
    assert set(sampled["contact_type"]) == {"cas"}
    assert np.allclose(sampled["weight"], 1.0)


def test_graph_sampling_is_seed_reproducible(
    toy_model_factory,
):
    config = toy_model_factory.config(
        population_overrides={
            "wp_contacts": 0,
            "sch_contacts": 0,
            "gq_contacts": 0,
            "cas_contacts": 2,
        },
        sim_overrides={
            "master_casual_candidates": 10,
        },
    )

    master_edges = pd.DataFrame(
        [
            (source, target, 0.2, "cas")
            for source in range(6)
            for target in range(source + 1, 6)
        ],
        columns=[
            "source",
            "target",
            "weight",
            "contact_type",
        ],
    )

    minimal = (
        build_minimal_graphdata_from_edge_list(
            master_edges,
            N=6,
        )
    )

    first = sample_from_master_graphdata(
        minimal,
        config,
        seed=44,
    )
    second = sample_from_master_graphdata(
        minimal,
        config,
        seed=44,
    )

    pd.testing.assert_frame_equal(
        _canonical_edges(first),
        _canonical_edges(second),
    )


def test_different_graph_seeds_can_change_sample(
    toy_model_factory,
):
    config = toy_model_factory.config(
        population_overrides={
            "wp_contacts": 0,
            "sch_contacts": 0,
            "gq_contacts": 0,
            "cas_contacts": 2,
        },
        sim_overrides={
            "master_casual_candidates": 10,
        },
    )

    master_edges = pd.DataFrame(
        [
            (source, target, 0.2, "cas")
            for source in range(6)
            for target in range(source + 1, 6)
        ],
        columns=[
            "source",
            "target",
            "weight",
            "contact_type",
        ],
    )

    minimal = (
        build_minimal_graphdata_from_edge_list(
            master_edges,
            N=6,
        )
    )

    signatures = set()

    for seed in range(1, 8):
        sampled = sample_from_master_graphdata(
            minimal,
            config,
            seed=seed,
        )
        canonical = _canonical_edges(sampled)

        signatures.add(
            tuple(
                canonical[
                    ["a", "b"]
                ].itertuples(
                    index=False,
                    name=None,
                )
            )
        )

    assert len(signatures) > 1