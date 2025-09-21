from langchain.callbacks.base import BaseCallbackHandler

class ReasoningTracker(BaseCallbackHandler):
    def __init__(self):
        self.steps = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.steps.append(f"Chain started with input: {inputs.get('question')}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.steps.append(f"LLM called with prompt: {prompts[0]}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.steps.append(f"Tool called with input: {input_str}")

    def on_chain_end(self, outputs, **kwargs):
        self.steps.append(f"Chain ended with outputs: {outputs}")

    def on_llm_end(self, response, **kwargs):
        self.steps.append(f"LLM returned response: {response}")

    def get_steps(self):
        return self.steps
