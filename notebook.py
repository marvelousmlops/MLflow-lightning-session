# Databricks notebook source
# MAGIC !pip install -e .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
from postai.model import SocialPoster
from postai.model_serving import ModelServing

os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="mlflow_genai", key="OPENAI_API_KEY")
os.environ["GEMINI_API_KEY"] = dbutils.secrets.get(scope="mlflow_genai", key="GEMINI_API_KEY")

# COMMAND ----------

system_prompt = """You are a social media content specialist with expertise in matching writing styles and voice across platforms. Your task is to:
1. Analyze the provided example post(s) by examining:
    - Writing style, tone, and voice
    - Sentence structure and length
    - Use of hashtags, emojis, and formatting
    - Engagement techniques and calls-to-action
2. Generate a new LinkedIn post about the given topic that matches:
    - The identified writing style and tone
    - Similar structure and formatting choices
    - Equivalent use of platform features and hashtags
    - Comparable engagement elements
3. Return only the generated post, formatted exactly as it would appear on LinkedIn, without any additional commentary or explanations."""

prompt_template = """
example posts:
{example_posts}
context:
{context}
additional instructions:
{additional_instructions}
"""

config = {
    "system_prompt": system_prompt,
    "prompt_template": prompt_template,
    "model_provider": "google",
    "model_name": "gemini-2.0-flash-exp",
}

# Instantiate the custom model
model = SocialPoster(config)


# COMMAND ----------

model_info = model.log_and_register_model(artifact_path="social_poster",
                                          model_name="mlflow_lightening_session.dev.social-ai")

# COMMAND ----------

post_example_1 = """MLflow's GenAI evaluation metrics now work as callable functions as of MLflow 2.17, making them easier to use and integrate.

Now you can use metrics like answer_relevance, answer_correctness, faithfulness, and toxicity directly as functions—no need to go through mlflow.evaluate() anymore if you're just prototyping with individual metrics or integrating metric calls into systems where mlflow.evalaute is not necessary.

This means:

🔍 Easier debugging during prototyping

🔌 More flexible integration options

🎯 Works with or without other MLflow features

Check it out in action ⬇️

Learn more:

📚 Docs: https://lnkd.in/gyBzcrDr

📝 Release notes: https://lnkd.in/gBrNQfFC

#MachineLearning #AI #LLMs #LLMOps #Evals"""

post_example_2 = """If you're already building with Python ML libraries, adding mlflow.autolog() to your code instantly gives you production-grade experiment tracking, model management, and observability—no extra infrastructure or code changes needed.

The automatic logging works across a remarkable breadth of libraries—from GenAI frameworks like LangChain, OpenAI, and LlamaIndex to traditional ML and deep learning libraries like PyTorch, scikit-learn, and Fastai.

MLflow's autolog feature changes this equation. With a single line—mlflow.autolog()—you get automatic logging of:

📊 Training metrics and parameters for scikit-learn, PyTorch, many other ML frameworks

🔄 LLM traces, prompts, responses, and tool calls for OpenAI and LangChain

📦 Model signatures and artifacts

💾 Dataset information and example inputs

The best part is that it works out of the box with the most popular libraries in the Python ML ecosystem: no need to modify your existing training code or add manual logging statements.

Read more: https://lnkd.in/e_aTp6HH

#machinelearning #mlops #ai #llmops"""

post_example_3 = """New tutorial: Step-by-step guide to building a tool-calling LLM application using MLflow's ChatModel wrapper and tracing system.

This tutorial shows you how to:

🔧 Create a tool-calling model using mlflow.pyfunc.ChatModel

🔄 Implement OpenAI function calling with automatic input/output handling

🔍 Add comprehensive tracing to debug multi-step LLM interactions

🚀 Deploy your model with full MLflow lifecycle management

The guide includes a practical example building a weather information agent, showing how ChatModel simplifies complex LLM patterns while providing enterprise-grade observability.

Check out the complete tutorial here: https://lnkd.in/gdTw8N2S

#MLOps #AIEngineering #LLMOps #AI"""


# COMMAND ----------

import mlflow

sample_input = [{
    "example_posts": ["Example 1: This is an example post.", "Example 2: This is another example post."],
    "context_url": "https://mlflow.org/docs/latest/llms/",
    "additional_instructions": "The post should be concise and to the point..."
}]

model_uri = f"models:/mlflow_lightening_session.dev.social-ai@latest-model"
model = mlflow.pyfunc.load_model(model_uri)
model.predict(sample_input)


# COMMAND ----------

endpoint_name = "social-ai"
model_server = ModelServing(
    model_name="mlflow_lightening_session.dev.social-ai",
    endpoint_name=endpoint_name,
)
model_server.deploy_or_update_serving_endpoint()

# COMMAND ----------

post_example_1 = """MLflow's GenAI evaluation metrics now work as callable functions as of MLflow 2.17, making them easier to use and integrate.

Now you can use metrics like answer_relevance, answer_correctness, faithfulness, and toxicity directly as functions—no need to go through mlflow.evaluate() anymore if you're just prototyping with individual metrics or integrating metric calls into systems where mlflow.evalaute is not necessary.

This means:

🔍 Easier debugging during prototyping

🔌 More flexible integration options

🎯 Works with or without other MLflow features

Check it out in action ⬇️

Learn more:

📚 Docs: https://lnkd.in/gyBzcrDr

📝 Release notes: https://lnkd.in/gBrNQfFC

#MachineLearning #AI #LLMs #LLMOps #Evals"""


post_example_2 = """If you're already building with Python ML libraries, adding mlflow.autolog() to your code instantly gives you production-grade experiment tracking, model management, and observability—no extra infrastructure or code changes needed.

The automatic logging works across a remarkable breadth of libraries—from GenAI frameworks like LangChain, OpenAI, and LlamaIndex to traditional ML and deep learning libraries like PyTorch, scikit-learn, and Fastai.

MLflow's autolog feature changes this equation. With a single line—mlflow.autolog()—you get automatic logging of:

📊 Training metrics and parameters for scikit-learn, PyTorch, many other ML frameworks

🔄 LLM traces, prompts, responses, and tool calls for OpenAI and LangChain

📦 Model signatures and artifacts

💾 Dataset information and example inputs

The best part is that it works out of the box with the most popular libraries in the Python ML ecosystem: no need to modify your existing training code or add manual logging statements.

Read more: https://lnkd.in/e_aTp6HH

#machinelearning #mlops #ai #llmops"""


post_example_3 = """New tutorial: Step-by-step guide to building a tool-calling LLM application using MLflow's ChatModel wrapper and tracing system.

This tutorial shows you how to:

🔧 Create a tool-calling model using mlflow.pyfunc.ChatModel

🔄 Implement OpenAI function calling with automatic input/output handling

🔍 Add comprehensive tracing to debug multi-step LLM interactions

🚀 Deploy your model with full MLflow lifecycle management

The guide includes a practical example building a weather information agent, showing how ChatModel simplifies complex LLM patterns while providing enterprise-grade observability.

Check out the complete tutorial here: https://lnkd.in/gdTw8N2S

#MLOps #AIEngineering #LLMOps #AI"""


example_posts = [post_example_1, post_example_2, post_example_3]


# COMMAND ----------

import os

os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

import requests

input = {"inputs":
         sample_input}

serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
response = requests.post(serving_endpoint, headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"}, json=input)
response.json()