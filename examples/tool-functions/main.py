import typing as t
import ollama

from ollama._toolfuncs import annotated_tool  # Pull Request coming soon


def lookup_function(function_name: str):
    return globals()[function_name]


# Example Function 1
@annotated_tool
def get_current_weather(city: t.Annotated[str, "The name of the city"]):
    # mock response
    return 75.0, "Farenheit"


def run_example_1():
    response_1 = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
        tools=[get_current_weather.tool_schema],
    )

    fn_call_1 = response_1["message"]["tool_calls"][0]["function"]
    fn_resp_1 = lookup_function(fn_call_1["name"])(city=fn_call_1["arguments"]["city"])
    assert fn_resp_1 == (75.0, "Farenheit")
    print("SUCCESS: get_current_weather is called")


run_example_1()


# Example Function 2
@annotated_tool
def sort_fruit_names(
    fruit_names: t.Annotated[t.List[str], "a list of fruit names"]
) -> t.List:
    return sorted(fruit_names)


def run_example_2():
    response_2 = ollama.chat(
        model="llama3.1",
        messages=[
            {
                "role": "user",
                "content": "Sort the fruit names strawberry, kiwi, banana, pineapple, passion fruit, mango",
            }
        ],
        tools=[
            get_current_weather.tool_schema,  # not expecting a match
            sort_fruit_names.tool_schema,  # expecting a match
        ],
    )
    print(response_2["message"]["tool_calls"])
    assert len(response_2["message"]["tool_calls"]) == 1
    fn_call_2 = response_2["message"]["tool_calls"][0]["function"]
    fruit_names_arg = fn_call_2["arguments"]["fruit_names"]
    print(f"model found {len(fruit_names_arg)} fruit names: {fruit_names_arg}")
    fn_resp_2 = lookup_function(fn_call_2["name"])(fruit_names=fruit_names_arg)
    print(fn_resp_2)
    assert fn_resp_2 == [
        "banana",
        "kiwi",
        "mango",
        "passion fruit",
        "pineapple",
        "strawberry",
    ]
    print("SUCCESS: sort_fruit_names is called")


run_example_2()
