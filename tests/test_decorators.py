import context
import mydecorators

def test_preprocess_say_hello():
    @mydecorators.preprocess('name',lambda s:s.capitalize())
    def greet(name):
        return f"Hello {name}"
    assert greet("joe") == "Hello Joe"

def test_singleOrList_noarg_with_list_input():
    @mydecorators.singleOrList
    def doubleAll(listInput):
        return [2*x for x in listInput]
    assert doubleAll([1,2]) == [2,4]

def test_singleOrList_noarg_with_single_input():
    @mydecorators.singleOrList
    def doubleAll(listInput):
        return [2*x for x in listInput]
    assert doubleAll(3) == [6]

def test_singleOrList_kw_with_list_input():
    @mydecorators.singleOrList(kw='listInput')
    def doubleAll(listInput):
        return [2*x for x in listInput]
    assert doubleAll([1,2]) == [2,4]
    assert doubleAll(listInput=[3,1]) == [6,2]

def test_singleOrList_kw_with_single_input():
    @mydecorators.singleOrList(kw='listInput')
    def doubleAll(listInput):
        return [2*x for x in listInput]
    assert doubleAll(3) == [6]
    assert doubleAll(listInput=3) == [6]

def test_singleOrList_kw_and_other_with_list_input():
    @mydecorators.singleOrList(kw='listInput')
    def doubleAll(foo,listInput):
        return [2*x for x in listInput]
    assert doubleAll(0,[1,2]) == [2,4]
    assert doubleAll(0,listInput=[3,1]) == [6,2]

def test_singleOrList_kw_and_other_with_single_input():
    @mydecorators.singleOrList(kw='listInput')
    def doubleAll(foo,listInput):
        return [2*x for x in listInput]
    assert doubleAll(0,3) == [6]
    assert doubleAll(0,listInput=3) == [6]
