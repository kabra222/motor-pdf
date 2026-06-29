from app.agent.store import PersistentVectorStore, add_message, create_session, get_history


def test_persistent_vector_store():
    store = PersistentVectorStore(namespace="test")
    store.clear()
    assert store.size == 0

    store.add("doc1", "texto de exemplo", [0.1, 0.2, 0.3], {"page": 1})
    assert store.size == 1

    results = store.search([0.1, 0.2, 0.3], top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["metadata"]["page"] == 1

    store.clear()
    assert store.size == 0


def test_vector_store_search_threshold():
    store = PersistentVectorStore(namespace="test_threshold")
    store.clear()
    store.add("a", "texto a", [1.0, 0.0, 0.0])
    store.add("b", "texto b", [0.0, 1.0, 0.0])

    results = store.search([1.0, 0.0, 0.0], top_k=5, threshold=0.5)
    assert len(results) == 1
    assert results[0]["id"] == "a"

    results = store.search([1.0, 0.0, 0.0], top_k=5, threshold=0.9)
    assert len(results) == 1
    assert results[0]["id"] == "a"

    store.clear()


def test_vector_store_namespace_isolation():
    store_a = PersistentVectorStore(namespace="ns_a")
    store_b = PersistentVectorStore(namespace="ns_b")
    store_a.clear()
    store_b.clear()

    store_a.add("x", "texto", [0.5, 0.5])
    assert store_a.size == 1
    assert store_b.size == 0

    store_a.clear()
    store_b.clear()


def test_session_workflow():
    session_id = create_session()
    assert session_id

    add_message(session_id, "user", "Ola")
    add_message(session_id, "assistant", "Oi!")

    history = get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Ola"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Oi!"
