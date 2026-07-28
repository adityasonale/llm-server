import mlflow

class GroqModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from providers.groq.services import GroqService
        self.service = GroqService()

    def predict(self, context, model_input):
        import asyncio
        prompt = model_input["prompt"][0]
        system = model_input.get("system", ["You are a helpful assistant."])[0]
        model = model_input.get("model", ["llama-3.3-70b-versatile"])[0]
        return asyncio.run(self.service.fetch_response(system, prompt, model))