from copilot.dataset import (
    SourceFile,
    cap_project,
    collect_project,
    count_tokens,
    skip_reason,
    split,
)


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_skip_rules():
    seen = set()
    assert skip_reason("tests/test_x.py", "x = 1\n", seen) == "test/docs/example"
    assert skip_reason("pkg/test_x.py", "x = 1\n", seen) == "test/docs/example"
    assert skip_reason("docs/conf.py", "x = 1\n", seen) == "test/docs/example"
    assert skip_reason("setup.py", "x = 1\n", seen) == "test/docs/example"
    assert skip_reason("pkg/__init__.py", "  \n", seen) == "empty"
    assert skip_reason("pkg/old.py", "print 'py2'\n", seen) == "doesn't parse"
    assert skip_reason("pkg/a.py", "x = 1\n", seen) is None
    assert skip_reason("other/b.py", "x = 1\n", seen) == "duplicate"
    # a file only called "testing_utils.py" isn't a test file
    assert skip_reason("pkg/testing_utils.py", "y = 2\n", seen) is None


def test_count_tokens_ignores_comments_and_blank_lines():
    assert count_tokens("x = 1  # hi\n\n") == count_tokens("x = 1\n")


def test_collect_project(tmp_path):
    proj = tmp_path / "proj"
    write(proj, "pkg/a.py", "def f():\n    return 1\n")
    write(proj, "pkg/b.py", "x = 2\n")
    write(proj, "tests/test_a.py", "assert True\n")
    write(proj, "README.md", "not python")
    kept, dropped = collect_project(proj, set())
    assert [f.path.split("proj/")[1] for f in kept] == ["pkg/a.py", "pkg/b.py"]
    assert all(f.project == "proj" for f in kept)
    assert dropped == {"test/docs/example": 1}


def fake_files(project, n, tokens=10):
    return [SourceFile(project, f"data/raw/{project}/f{i}.py", tokens) for i in range(n)]


def test_cap_leaves_small_projects_alone():
    files = fake_files("small", 5)
    assert cap_project(files, 1000, seed=1) == files


def test_cap_is_under_limit_and_repeatable():
    files = fake_files("big", 100)
    capped = cap_project(files, 255, seed=1)
    assert sum(f.tokens for f in capped) <= 255
    assert len(capped) == 25
    assert capped == cap_project(files, 255, seed=1)
    assert capped != cap_project(files, 255, seed=2)


def many_projects():
    files = []
    for i, size in enumerate([5, 8, 3, 10, 6, 7, 4, 9, 5, 6]):
        files += fake_files(f"p{i}", size)
    return files


def test_split_by_project_keeps_projects_on_one_side():
    files = many_projects()
    train, test = split(files, "project", 0.3, seed=329)
    train_projects = {f.project for f in train}
    test_projects = {f.project for f in test}
    assert train_projects and test_projects
    assert not train_projects & test_projects
    assert len(train) + len(test) == len(files)
    test_share = sum(f.tokens for f in test) / sum(f.tokens for f in files)
    assert 0.2 <= test_share <= 0.4


def test_split_by_file():
    files = many_projects()
    train, test = split(files, "file", 0.3, seed=329)
    assert not {f.path for f in train} & {f.path for f in test}
    assert len(test) == round(len(files) * 0.3)


def test_split_is_repeatable():
    files = many_projects()
    assert split(files, "project", 0.3, 1) == split(files, "project", 0.3, 1)
    assert split(files, "file", 0.3, 1) == split(files, "file", 0.3, 1)
