import textwrap

from src.chunker import chunk_directory, chunk_source

MULTI_CLASS_SOURCE = textwrap.dedent(
    """\
    class Alpha:
        def method_one(self):
            return 1

        def method_two(self):
            return 2


    class Beta:
        async def method_three(self):
            return 3


    def standalone_function():
        return 4
    """
)


def test_chunk_source_splits_on_function_and_class_boundaries():
    chunks = chunk_source(MULTI_CLASS_SOURCE, "fixture.py")
    qualified_names = {c.qualified_name for c in chunks}

    assert qualified_names == {
        "Alpha",
        "Alpha.method_one",
        "Alpha.method_two",
        "Beta",
        "Beta.method_three",
        "standalone_function",
    }


def test_chunk_source_assigns_correct_kind_per_boundary():
    chunks = {c.qualified_name: c for c in chunk_source(MULTI_CLASS_SOURCE, "fixture.py")}

    assert chunks["Alpha"].kind == "class"
    assert chunks["Alpha.method_one"].kind == "method"
    assert chunks["Beta.method_three"].kind == "async_method"
    assert chunks["standalone_function"].kind == "function"


def test_chunk_source_line_ranges_match_real_boundaries():
    chunks = {c.qualified_name: c for c in chunk_source(MULTI_CLASS_SOURCE, "fixture.py")}
    lines = MULTI_CLASS_SOURCE.splitlines()

    method_one = chunks["Alpha.method_one"]
    assert lines[method_one.start_line - 1].strip().startswith("def method_one")
    assert lines[method_one.end_line - 1].strip() == "return 1"

    alpha = chunks["Alpha"]
    assert lines[alpha.start_line - 1].strip() == "class Alpha:"
    # the class chunk's span must cover both of its methods
    assert alpha.start_line <= method_one.start_line
    assert alpha.end_line >= chunks["Alpha.method_two"].end_line


def test_chunk_source_falls_back_to_line_chunking_on_syntax_error(caplog):
    broken_source = "def broken(:\n    pass\n"

    with caplog.at_level("WARNING"):
        chunks = chunk_source(broken_source, "broken.py")

    assert len(chunks) == 1
    assert chunks[0].kind == "line_fallback"
    assert "falling back to line chunking" in caplog.text


def test_chunk_directory_walks_real_corpus():
    chunks = chunk_directory("corpus")
    qualified_names = {c.qualified_name for c in chunks}

    assert "factorial" in qualified_names
    assert "reverse_string" in qualified_names
    assert all(c.kind in {"function", "async_function", "method", "async_method", "class"}
               for c in chunks)
