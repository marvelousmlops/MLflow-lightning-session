import os
import requests
from markdownify import markdownify
import mlflow
from mlflow.pyfunc import PythonModel
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from mlflow import MlflowClient
from mlflow.models import set_model
from mlflow.exceptions import MlflowException

class SocialPostInput(BaseModel):
    example_posts: List[str]
    context_url: str
    additional_instructions: str


class SocialPostOutput(BaseModel):
    post: str


class SocialPoster(PythonModel):
    def __init__(self, config=None):
        default_config = {
            "system_prompt": (
                "You are a social media content specialist with expertise in matching writing "
                "styles and voice across platforms. Your task is to:\n\n"
                "1. Analyze the provided example post(s) by examining:\n"
                "   - Writing style, tone, and voice\n"
                "   - Sentence structure and length\n\n"
                "2. Return only the generated post\n"
            ),
            
            "prompt_template": (
                "example posts:"
                "{example_posts}"
                "context:"
                "{context}"
                "additional instructions:"
                "{additional_instructions}"
            ),
            
            "model_provider": "google",
            "model_name": "gemini-2.0-flash-exp",
        }
        self.config = config if config else default_config
        self.tracing_enabled = False
        self.mlflow_client = MlflowClient()

    @mlflow.trace(span_type="FUNCTION")
    def _webpage_to_markdown(self, url):
        response = requests.get(url)
        html_content = response.text
        markdown_content = markdownify(html_content)
        return markdown_content

    @mlflow.trace(span_type="FUNCTION")
    def _generate_prompt(self, example_posts, context, additional_instructions):
        example_posts = "\n".join(
            [f"Example {i+1}:\n{post}" for i, post in enumerate(example_posts)]
        )
        prompt = self.config["prompt_template"].format(
            example_posts=example_posts,
            context=context,
            additional_instructions=additional_instructions,
        )
        formatted_prompt = [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": prompt},
        ]
        return formatted_prompt

    @mlflow.trace(span_type="LLM")
    def _generate_post(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def load_context(self, context):
        self.system_prompt = context.model_config["system_prompt"]
        self.prompt_template = context.model_config["prompt_template"]
        self.model_provider = context.model_config["model_provider"]
        self.model_name = context.model_config["model_name"]
        self.tracing_enabled = os.getenv("MLFLOW_TRACING_ENABLED", "false").lower() == "true"

        if self.model_provider == "openai":
            self.client = OpenAI()
        elif self.model_provider == "google":
            self.client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.environ["GEMINI_API_KEY"]
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def predict(self, context, model_input: list[SocialPostInput]) -> list[SocialPostOutput]:
        model_input = model_input[0].model_dump()
        if not mlflow.tracing.provider.is_tracing_enabled() == self.tracing_enabled:
            mlflow.tracing.enable() if self.tracing_enabled else mlflow.tracing.disable()

        with mlflow.start_span(name="predict", span_type="CHAIN") as parent_span:
            parent_span.set_inputs(model_input)
            example_posts = model_input.get("example_posts")
            context_url = model_input.get("context_url")
            markdown_context = self._webpage_to_markdown(context_url)
            additional_instructions = model_input.get("additional_instructions")

            prompt = self._generate_prompt(example_posts, markdown_context, additional_instructions)
            post = self._generate_post(prompt)
            parent_span.set_outputs({"post": post})
        return [{"post": post}]

    def log_and_register_model(self, artifact_path, model_name):
        pip_requirements = [
        "openai",
        "markdownify",
        ]

        try:
            self.mlflow_client.get_registered_model(model_name)
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                self.mlflow_client.create_registered_model(model_name)
            else:
                raise

        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                artifact_path,
                python_model=self,
                model_config=self.config,
                pip_requirements=pip_requirements,
            )

            mv = self.mlflow_client.create_model_version(
            name=model_name,
            source=model_info.model_uri)

            self.mlflow_client.set_registered_model_alias(
            name=model_name,
            alias="latest-model",
            version=mv.version,
                )
        return model_info
    
set_model(SocialPoster())