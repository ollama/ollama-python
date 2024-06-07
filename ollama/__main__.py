from absl import app, flags

from ollama._client import Client


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "endpoint", None, "URL for the ollama endpoint", short_name="e", required=True
)
flags.DEFINE_string(
    "model", None, "Name of the model to run", short_name="m", required=True
)
flags.DEFINE_string(
    "prompt", None, "Text prompt submitted to the model", short_name="p", required=False
)


def main(argv):
    del argv
    endpoint = FLAGS.endpoint
    model = FLAGS.model
    prompt = FLAGS.prompt

    if prompt is not None:
        client = Client(endpoint)
        r = client.generate(model, prompt)
        print("\n[[", r["response"], "]]")
        return


if __name__ == "__main__":
    app.run(main)
