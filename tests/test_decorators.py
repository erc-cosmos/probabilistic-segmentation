"""Testing for decorators."""
import my_decorators


def test_preprocess_say_hello():
    """Test that preprocess actually preprocesses a keyword argument."""
    @my_decorators.preprocess('name', lambda s: s.capitalize())
    def greet(name):
        return f"Hello {name}"
    assert greet("joe") == "Hello Joe"


def test_single_or_list_noarg_with_list_input():
    """Test that single_or_list with no argument leaves a list alone."""
    @my_decorators.single_or_list
    def double_all(list_input):
        return [2*x for x in list_input]
    assert double_all([1, 2]) == [2, 4]


def test_single_or_list_noarg_with_single_input():
    """Test that single_or_list with no argument turns a scalar into a list."""
    @my_decorators.single_or_list
    def double_all(list_input):
        return [2*x for x in list_input]
    assert double_all(3) == [6]


def test_single_or_list_kw_with_list_input():
    """Test that single_or_list with a keyword argument leaves a list alone."""
    @my_decorators.single_or_list(kw='list_input')
    def double_all(list_input):
        return [2*x for x in list_input]
    assert double_all([1, 2]) == [2, 4]
    assert double_all(list_input=[3, 1]) == [6, 2]


def test_single_or_list_kw_with_single_input():
    """Test that single_or_list with a keyword argument turns a scalar into a list."""
    @my_decorators.single_or_list(kw='list_input')
    def double_all(list_input):
        return [2*x for x in list_input]
    assert double_all(3) == [6]
    assert double_all(list_input=3) == [6]


def test_single_or_list_kw_and_other_with_list_input():
    """Test that single_or_list with a keyword argument leaves a list alone and leaves the rest alone."""
    @my_decorators.single_or_list(kw='list_input')
    def double_all(foo, list_input):
        return [2*x for x in list_input]
    assert double_all(0, [1, 2]) == [2, 4]
    assert double_all(0, list_input=[3, 1]) == [6, 2]


def test_single_or_list_kw_and_other_with_single_input():
    """Test that single_or_list with a keyword argument turns a scalar into a list and leaves the rest alone."""
    @my_decorators.single_or_list(kw='list_input')
    def double_all(foo, list_input):
        return [2*x for x in list_input]
    assert double_all(0, 3) == [6]
    assert double_all(0, list_input=3) == [6]
