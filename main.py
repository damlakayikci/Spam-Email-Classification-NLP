import json


def read_data(file):
    file = open(file)
    data = json.load(file)
    return data


if __name__ == "__main__":

    read_data("Oppositional_thinking_analysis_dataset.json")