import operator
import os
import random
import shutil
from math import isclose

import faiss
import numpy as np
import pytest
from sklearn.metrics import precision_score
from sklearn.metrics.pairwise import cosine_similarity

from mellow_db.exceptions import ResourceNotFoundError
from mellow_db.storage import gcs_bucket_connection
from mellow_db.vector_db import FaissIndex


@pytest.mark.parametrize("type_key", [("type_1"), ("type_2")])
def test_index_object(embeddings, faiss_urls, type_key):
    index_object = FaissIndex(db_url=faiss_urls[type_key])
    assert isinstance(index_object, FaissIndex)
    assert index_object.index_ is None
    index_object.add(embeddings[type_key]).commit()
    assert isinstance(index_object, FaissIndex)
    assert isinstance(index_object.index_, faiss.swigfaiss_avx2.IndexFlat)
    assert len(index_object.get_embeddings()) == len(embeddings[type_key])
    assert len(index_object.get_keys()) == len(embeddings[type_key])


def test_add_keys_embeddings_count(embeddings, faiss_urls):
    index = FaissIndex(db_url=faiss_urls["type_1"])
    keys = list(embeddings["type_1"].keys())[:150]
    data = {k: embeddings["type_1"][k] for k in keys}

    batch1 = {key: data[key] for key in keys[:80]}
    index.add(batch1).commit()
    assert index.get_count() == 80
    assert len(index.get_embeddings()) == 80
    assert set(index.idx_to_key.keys()) == set(range(80))
    assert index.get_keys() == keys[:80]
    assert np.array_equal(np.array(index.get_embeddings(reshape=None)),
                          np.array(list(batch1.values())))

    batch2 = {key: data[key] for key in keys[80:140]}
    index.add(batch2).commit()
    assert index.get_count() == 140
    assert len(index.get_embeddings()) == 140
    assert set(index.idx_to_key.keys()) == set(range(140))
    assert index.get_keys() == keys[:140]
    assert np.array_equal(np.array(index.get_embeddings(reshape=None)),
                          np.array(list({**batch1, **batch2}.values())))

    batch3 = {key: data[key] for key in keys[140:]}
    index.add(batch3).commit()
    assert index.get_count() == 150
    assert len(index.get_embeddings()) == 150
    assert set(index.idx_to_key.keys()) == set(range(150))
    assert index.get_keys() == keys
    assert np.array_equal(np.array(index.get_embeddings(reshape=None)),
                          np.array(list({**batch1, **batch2, **batch3}.values())))

    batch_overwrite = {key: data[key] for key in random.sample(keys, 50)}
    index.add(batch_overwrite, upsert=True).commit()
    assert index.get_count() == 150
    assert len(index.get_embeddings()) == 150
    assert set(index.idx_to_key.keys()) == set(range(150))
    assert index.get_keys() == keys
    assert np.array_equal(np.array(index.get_embeddings(reshape=None)),
                          np.array(list({**batch1, **batch2, **batch3}.values())))


@pytest.mark.parametrize("key, exists", [
    ('100', True),
    ('102', True),
    ('abc', False),
    (1, False),
    ('', False),
    (None, False),
])
def test_key_exists(indexes, key, exists):
    assert indexes["type_1"].key_exists(key) is exists


@pytest.mark.parametrize("type_key, key, reshape, expected_shape, exception", [
    ('type_1', '103', (1, -1), (1, 768), False),
    ('type_1', '103', (-1, 1), (768, 1), False),
    ('type_2', '211', (1, -1), (1, 512), False),
    ('type_1', '103', (2, -1), (2, 384), False),
    ('type_1', 123, (1, -1), None, KeyError),
    ('type_1', 'abc', (1, -1), None, KeyError),
])
def test_get_embedding(indexes, embeddings, type_key, key, reshape, expected_shape, exception):
    if exception:
        with pytest.raises(exception):
            emb = indexes[type_key].get_embedding(key=key, reshape=reshape)
    else:
        emb = indexes[type_key].get_embedding(key=key, reshape=reshape)
        assert emb.shape == expected_shape
        assert np.array_equal(emb.reshape(reshape), embeddings[type_key][key].reshape(reshape))


@pytest.mark.parametrize("type_key, n_samples, n_results", [
    ("type_1", 1, 20), ("type_1", 10, 30), ("type_2", 1, 50)
])
def test_search_basic(embeddings, indexes, type_key, n_samples, n_results):
    keys = list(embeddings[type_key].keys())
    search_keys = random.sample(keys, n_samples)
    search_res = indexes[type_key].search(search_keys, n_results=n_results)
    assert len(search_res) == n_samples
    assert all(len(res_item) == n_results for res_item in search_res)
    assert all(-1.001 <= res[1] <= 1.001 for res_item in search_res for res in res_item)


@pytest.mark.parametrize("n_query, n_subset", [
    (10, 250), (100, 50), (3, 5)
])
def test_search_subset(embeddings, indexes, n_query, n_subset):
    keys = list(embeddings["type_1"].keys())
    search_keys = random.sample(keys, n_query)
    subset_keys = random.sample(list(set(keys) - set(search_keys)), n_subset)
    search_res = indexes["type_1"].search(search_keys, subset_keys=subset_keys, n_results=n_subset)
    assert set(res[0] for res_item in search_res for res in res_item) == set(subset_keys)
    assert all(-1.001 <= res[1] <= 1.001 for res_item in search_res for res in res_item)


@pytest.mark.parametrize("n_samples, n_results, projection, return_fields", [
    (1, 20, ["key"], False),
    (10, 5, ["similarity"], False),
    (8, 5, ["similarity", "key"], False),
    (10, 5, ["similarity"], True),
    (8, 5, ["key", "similarity"], True),
])
def test_search_projection(embeddings, indexes, n_samples, n_results, projection, return_fields):
    keys = list(embeddings["type_1"].keys())
    search_keys = random.sample(keys, n_samples)
    search_res = indexes["type_1"].search(
        search_keys, n_results=n_results, projection=projection, return_field_names=return_fields)
    assert len(search_res) == n_samples
    assert all(len(res_item) == n_results for res_item in search_res)
    assert all(len(res) == len(projection) for res_item in search_res for res in res_item)
    if "similarity" in projection and not return_fields:
        similarity_index = projection.index("similarity")
        assert all(-1.001 <= res[similarity_index] <= 1.001 for res_item in search_res for res in res_item)
    if return_fields:
        assert all(res.get(key, None) is not None for res_item in search_res for res in res_item for key in projection)
        if "similarity" in projection:
            assert all(-1.001 <= res.get("similarity") <= 1.001 for res_item in search_res for res in res_item)


@pytest.mark.parametrize("search_keys, n_results, expected", [
    ("1", 20, [type(None)]),
    (["1", "5", "7"], 50, [type(None), type(None), type(None)]),
    ([2, 3, 4, 6], 30, [type(None), type(None), type(None), type(None)]),
    ([2, '100', 4, '102'], 30, [type(None), list, type(None), list]),
    (["101", "abc"], 30, [list, type(None)]),
])
def test_search_none_cases(indexes, search_keys, n_results, expected):
    index_ = indexes["type_1"]
    with pytest.raises(KeyError):
        index_.search(search_keys, n_results=n_results, not_exists_ok=False)
    search_res = index_.search(search_keys, n_results=n_results, not_exists_ok=True, projection=["similarity"])
    assert list(map(type, search_res)) == expected
    for i, exp in enumerate(expected):
        if exp == list:
            assert len(search_res[i]) == n_results
            assert all(-1.001 <= res[0] <= 1.001 for res in search_res[i])


@pytest.mark.parametrize("n_samples, n_results", [(10, 40), (50, 100)])
def test_search_batch(embeddings, indexes, n_samples, n_results):
    keys = list(embeddings["type_1"].keys())
    search_keys = random.sample(keys, n_samples)
    single_search = []
    for key in search_keys:
        single_search.extend(indexes["type_1"].search(key, n_results=n_results))
    batch_search = indexes["type_1"].search(search_keys, n_results=n_results)
    for ix, search_res in enumerate(single_search):
        # check if found similar keys are very similar between single search and batch search
        assert precision_score(
            [v[0] for v in search_res], [v[0] for v in batch_search[ix]], average='macro', zero_division=0) >= 0.95
        # check if found similarity scores are close between single search and batch search
        assert all(isclose(v1[1], v2[1], rel_tol=1e-5) for v1, v2 in zip(search_res, batch_search[ix]))


@pytest.mark.parametrize("type_key, top_n, score, accept_prec_or_diff", [
    ("type_1", 20, 0.85, False),
    ("type_2", 50, 0.7, True)
])
def test_search_performance(embeddings, indexes, type_key, top_n, score, accept_prec_or_diff):
    keys = list(embeddings[type_key].keys())
    random_key = random.choice(keys)
    fr = indexes[type_key].search(random_key, n_results=100)[0]
    cr = cosine_similarity(embeddings[type_key][random_key].reshape(1, -1),
                           np.array(list(embeddings[type_key].values())))
    cr = {keys[ix]: cr[0][ix] for ix in np.argsort(cr[0])[::-1][:100]}

    diff_in_100 = sum(abs(a - b) > 1e-6
                      for a, b in zip(list(cr.values()), [item[1] for item in fr]))
    diff_in_top_n = sum(abs(a - b) > 1e-6
                        for a, b in zip(list(cr.values())[:top_n], [item[1] for item in fr][:top_n]))

    assert (operator.or_ if accept_prec_or_diff else operator.and_)(
        diff_in_100 < 2,
        precision_score(
            list(cr.keys()), [item[0] for item in fr], average='macro', zero_division=0) >= score
    )
    assert (operator.or_ if accept_prec_or_diff else operator.and_)(
        diff_in_top_n < 2,
        precision_score(
            list(cr.keys())[:top_n], [item[0] for item in fr][:top_n], average='macro', zero_division=0) >= score
    )


def test_make_index(indexes, embeddings, faiss_urls_with_data):
    temp_idx_to_key, temp_index = FaissIndex(db_url=faiss_urls_with_data["type_1"])._make_index(embeddings["type_1"])
    assert temp_idx_to_key == indexes["type_1"].idx_to_key
    FaissIndex.index_property_equals(temp_index, indexes["type_1"].index_)


def test_serialiazation(indexes, faiss_urls):
    data = indexes["type_1"].serialize()
    temp_index = FaissIndex.load_from_data(
        db_url=faiss_urls["type_1"], data=data["key_to_data"]).clone(db_url=faiss_urls["type_2"])
    FaissIndex.equals(temp_index, indexes["type_1"])


def test_commit(embeddings, faiss_urls):
    index = FaissIndex(db_url=faiss_urls["type_1"])
    keys = list(embeddings["type_1"].keys())[:150]
    data = {k: embeddings["type_1"][k] for k in keys}
    assert index.key_exists(keys[10]) is False

    batch1 = {key: data[key] for key in keys[:80]}
    index.add(batch1)
    assert index.get_count() == 0
    assert index.key_exists(keys[10]) is False
    assert index.commit().get_count() == 80
    assert index.key_exists(keys[10]) is True
    assert index.key_exists(keys[100]) is False

    batch2 = {key: data[key] for key in keys[80:]}
    index.add(batch2)
    assert index.get_count() == 80
    assert index.key_exists(keys[100]) is False
    assert index.commit().get_count() == 150
    assert index.key_exists(keys[100]) is True

    batch_overwrite = {key: data[key] for key in random.sample(keys, 50)}
    index.add(batch_overwrite, upsert=True)
    assert index.get_count() == 150
    assert index.key_exists(keys[100]) is True
    assert index.commit().get_count() == 150
    assert index.key_exists(keys[100]) is True

    with pytest.raises(ValueError):
        index.commit()  # nothing to commit

    index.commit(allow_empty_commit=True)  # nothing to commit


def test_rollback(embeddings, faiss_urls):
    index = FaissIndex(db_url=faiss_urls["type_1"])
    keys = list(embeddings["type_1"].keys())[:150]
    data = {k: embeddings["type_1"][k] for k in keys}
    assert index.key_exists(keys[10]) is False

    batch1 = {key: data[key] for key in keys[:80]}
    index.add(batch1).commit()
    assert index.get_count() == 80

    temp_index = index.clone(db_url=faiss_urls["type_1"])
    batch2 = {key: data[key] for key in keys[80:]}
    index.add(batch2)
    assert index.get_count() == 80
    with pytest.raises(AssertionError):
        # index has previous state
        FaissIndex.equals(index, temp_index)
    index.rollback()  # clear previous state
    FaissIndex.equals(index, temp_index)

    with pytest.raises(ValueError):
        index.commit()  # nothing to commit


def test_drop(indexes, faiss_urls):
    index = indexes["type_1"].clone(faiss_urls["type_1"])
    index.drop(not_exists_ok=True)
    with pytest.raises(ResourceNotFoundError):
        index.drop(not_exists_ok=False)


@pytest.mark.parametrize("test_input, expected_output", [
    (["123", "456"], ["123", "456"]),
    ([123, 456], [123, 456]),
    ("123", ["123"]), (123, [123])
])
def test_format_keys(test_input, expected_output):
    assert FaissIndex._format_keys(test_input) == expected_output


def test_get_info(indexes):
    info_ = indexes["type_1"].get_info()
    assert isinstance(info_['size_in_bytes'], int)
    del info_['size_in_bytes']
    assert info_ == {
        'faiss_index_type': 'Flat',
        'faiss_index_metric': 'METRIC_INNER_PRODUCT',
        'embedding_dim': 768,
        'item_count': 1000,
    }


# this will not run periodically with ci/cd pipelines
def test_back_up_and_load(indexes, base_dir, faiss_urls):
    # back up
    index1 = indexes["type_1"]
    backup_dir = os.path.join(base_dir, "temp_vector_db_backup")
    index1.back_up(backup_dir)
    assert os.path.exists(os.path.join(backup_dir, "vector_db.db"))

    # load from backup
    loaded_index1 = FaissIndex.load_from_path(
        os.path.join(backup_dir, "vector_db.db"),
        faiss_urls["type_1"])
    assert isinstance(loaded_index1, FaissIndex)
    assert isinstance(loaded_index1.index_, faiss.swigfaiss_avx2.IndexFlatIP)
    assert loaded_index1.key_to_data[random.choice(loaded_index1.get_keys())].shape == (768,)
    assert sorted(list(loaded_index1.key_to_data.keys())) == sorted(list(loaded_index1.idx_to_key.values()))
    shutil.rmtree(backup_dir)

    # test error cases
    with open(backup_dir, "w") as f:
        f.write("dummy text")
    with pytest.raises(NotADirectoryError):
        index1.back_up(backup_dir)
    os.remove(backup_dir)
    with pytest.raises(FileNotFoundError):
        loaded_index1 = FaissIndex.load_from_path(
            os.path.join(backup_dir, "vector_db.db"),
            faiss_urls["type_1"])


def test_back_up_to_gcs_and_load_from_gcs(gcp_creds, gcs_bucket, base_dir, indexes, faiss_urls):
    # back up
    bucket = gcs_bucket_connection(gcp_creds, gcs_bucket)
    folder_path = os.path.join(base_dir, "temp_vector_db_backup")
    path = os.path.join(folder_path, "vector_db.db")
    blob = bucket.blob(path)
    assert not blob.exists()
    indexes["type_1"].back_up_to_gcs(gcp_creds, gcs_bucket, folder_path)
    assert blob.exists()

    # load from backup
    loaded_index1 = FaissIndex.load_from_gcs(gcp_creds, gcs_bucket, path, faiss_urls["type_1"])
    assert isinstance(loaded_index1, FaissIndex)
    assert isinstance(loaded_index1.index_, faiss.swigfaiss_avx2.IndexFlatIP)
    assert loaded_index1.key_to_data[random.choice(loaded_index1.get_keys())].shape == (768,)
    assert sorted(list(loaded_index1.key_to_data.keys())) == sorted(list(loaded_index1.idx_to_key.values()))
    blob.delete()

    # test error cases
    dummy_blob = bucket.blob(folder_path)
    dummy_blob.upload_from_string("dummy text")
    with pytest.raises(ValueError):
        indexes["type_1"].back_up_to_gcs(gcp_creds, gcs_bucket, folder_path)
    dummy_blob.delete()
    with pytest.raises(ResourceNotFoundError):
        FaissIndex.load_from_gcs(gcp_creds, gcs_bucket, path, faiss_urls["type_1"])
