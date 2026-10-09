# pyright: reportMissingImports=false
import os
import threading
import requests
import json
try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    AzureOpenAI = None
    OpenAI = None
import time
#from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider
def AzureCliCredential():
    pass
def ChainedTokenCredential():
    pass
def DefaultAzureCredential():
    pass
def get_bearer_token_provider():
    pass
try:
    import google.generativeai as genai
except ImportError:
    genai = None

DEFAULT_PROXY_BASE_URL = "http://35.220.164.252:3888/v1/chat/completions"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1/chat/completions"
NVIDIA_DEFAULT_MODEL = "azure/openai/gpt-4o"
SHARE_KEY_PATH = os.path.join(os.path.dirname(__file__), "share_key.txt")
LEGACY_MODEL_NAMES = {"chatgpt", "llama", "gemini", "deepseek"}
#openai/openai/gpt-5.4
#azure/anthropic/claude-sonnet-5
#nvidia/deepseek-ai/deepseek-v4-flash
#azure/openai/gpt-4o
#nvidia/qwen/qwen3.5-35b-a3b
#gcp/google/gemini-2.5-flash

def _config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        value = config
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value
    if hasattr(config, "get"):
        try:
            return config.get(key, default=default)
        except TypeError:
            return config.get(key, default)
    return default


def _strip_response_signatures(response_json):
    if not isinstance(response_json, dict):
        return response_json
    for choice in response_json.get("choices", []):
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        for block in message.get("thinking_blocks", []) or []:
            if isinstance(block, dict) and "signature" in block:
                block["signature"] = ""
        provider_fields = message.get("provider_specific_fields", {}) or {}
        for block in provider_fields.get("thinking_blocks", []) or []:
            if isinstance(block, dict) and "signature" in block:
                block["signature"] = ""
    return response_json


def _load_share_key():
    if not os.path.exists(SHARE_KEY_PATH):
        return None
    with open(SHARE_KEY_PATH, "r") as f:
        key = f.read().strip()
    return key or None


class LLM:
    def __init__(self, model=f"nvidia,{NVIDIA_DEFAULT_MODEL}", config=None):
        
        print(f'using model: {model}')
        self.model_choice = model
        self.config = config
        self.t = _config_get(self.config, 'model.temperature', default=None)
        self.share_api_key = _load_share_key()
        self.proxy_provider = None
        if ',' in model:
            self.proxy_provider, self.model_choice = model.split(',', 1)
            self.proxy_provider = self.proxy_provider.strip().lower()
            self.model_choice = self.model_choice.strip()
            self.chat = self.proxy_chat
        elif model not in LEGACY_MODEL_NAMES:
            self.proxy_provider = "nvidia"
            self.model_choice = model
            self.chat = self.proxy_chat
        else:
            self.model = self._init_model(model)
            self.chat = self._init_chat(model)
        print('model choice:',self.model_choice,'temperature:',self.t)
        self.input_tokens = 0
        self.output_tokens = 0
        self._token_lock = threading.Lock()
        self._query_usage = threading.local()

    @property
    def last_usage(self):
        """Token counts for the last call on the current thread."""
        return getattr(self._query_usage, 'tokens', (0, 0))

    def proxy_chat(self,content):
        self._query_usage.tokens = (0, 0)
        provider = (self.proxy_provider or "nvidia").lower()
        use_openrouter = provider in {"openrouter", "or"}
        use_nvidia = provider in {"nvidia", "nv"}
        if use_openrouter:
            base_url = _config_get(
                self.config,
                "model.openrouter_base_url",
                default=OPENROUTER_BASE_URL,
            )
        elif use_nvidia:
            base_url = _config_get(
                self.config,
                "model.nvidia_base_url",
                default=NVIDIA_BASE_URL,
            )
        else:
            base_url = _config_get(
                self.config,
                "model.proxy_base_url",
                default=DEFAULT_PROXY_BASE_URL,
            )

        if use_nvidia:
            api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("MY_API_KEY")
        elif use_openrouter:
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("MY_API_KEY")
        else:
            api_key = os.getenv("MY_API_KEY")
        if not api_key:
            if use_nvidia:
                env_name = "NVIDIA_API_KEY or MY_API_KEY"
            elif use_openrouter:
                env_name = "OPENROUTER_API_KEY or MY_API_KEY"
            else:
                env_name = "MY_API_KEY"
            raise ValueError(f"Missing API key. Please set {env_name}.")
        original_api_key = api_key
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"  
        }
        
        if self.t is not None:
            data = {
                "model": self.model_choice, # 可以替换为需要的模型
                "messages": [
                    {"role": "user", "content": content}
                ],
                
                "temperature": self.t # 自行修改温度等参数
            }
        else:
            data = {
                "model": self.model_choice, # 可以替换为需要的模型
                "messages": [
                    {"role": "user", "content": content}
                ],
                
            }
        # Long completions (e.g. full GPU kernels) get cut at the gateway's
        # default output cap unless max_tokens is sent explicitly; a truncated
        # response has no closing </candidate> tag and parses to nothing.
        # Unset (default) sends no field -> existing behaviour unchanged.
        max_tokens = _config_get(self.config, 'model.max_tokens', default=None)
        if max_tokens:
            data["max_tokens"] = int(max_tokens)
        model_name = (self.model_choice or "").lower()
        if use_nvidia:
            # NVIDIA's OpenAI-compatible endpoint rejects the OpenRouter-style
            # `reasoning` field for GPT models. For these models, reasoning is
            # disabled by omitting that parameter entirely.
            if "claude" in model_name or "anthropic" in model_name:
                data["thinking"] = {"type": "disabled"}
            elif "qwen" in model_name or "deepseek" in model_name:
                data["chat_template_kwargs"] = {"enable_thinking": False}
            elif "gemini" in model_name:
                data["reasoning_effort"] = "none"
        elif use_openrouter:
            # OpenRouter reasoning.effort options:
            # - "none": disable reasoning entirely; some endpoints reject this because reasoning is mandatory.
            # - "minimal": lowest enabled reasoning effort, about 10% of max tokens.
            # - "low": light reasoning, about 20% of max tokens.
            # - "medium": balanced reasoning, about 50% of max tokens.
            # - "high": deep reasoning, about 80% of max tokens.
            # - "xhigh": maximum reasoning, about 95% of max tokens; Google models map this down to "high".
            # For Gemini/OpenRouter, use "minimal" as the lowest safe option instead of "none".
            reasoning_effort = _config_get(self.config, "model.reasoning_effort", default="minimal")
            data["reasoning"] = {"effort": reasoning_effort}
            site_url = _config_get(self.config, "model.openrouter_site_url", default=None)
            app_name = _config_get(self.config, "model.openrouter_app_name", default=None)
            if site_url:
                headers["HTTP-Referer"] = site_url
            if app_name:
                headers["X-Title"] = app_name
        tried_share_key = False
        consecutive_failures = 0
        # optional circuit breaker: a permanently failing endpoint (e.g. a
        # wrong model name -> HTTP 403 forever) previously spun for days;
        # with this cap the run dies loudly instead of zombieing.
        max_failures = _config_get(self.config, 'model.max_consecutive_llm_failures', default=None)
        while True:
            try:
                response = requests.post(
                    base_url,
                    headers=headers,
                    json=data,
                    timeout=_config_get(self.config, "model.timeout", default=300),
                )
                response.raise_for_status()
                response_json = response.json()
                response_json = _strip_response_signatures(response_json)
                output = response_json['choices'][0]["message"]["content"]
                
                if output is None or len(output) == 0:
                    print(f"Response content is empty, response: {response_json}")
                    with open("response.json", "w") as f:
                        json.dump(response_json, f,indent=4)
                    continue
                if response.status_code != 200:
                    print(f"Request failed with status code {response.status_code}")
                    try:
                        print("Response JSON:", response.json())
                    except Exception as e:
                        print(f"Error getting response JSON: {e}")
                        print("Response text:", response.text)
                else:
                    break
            except Exception as e:
                # Context overflow is permanent for THIS prompt: no key or wait
                # fixes it. Give up on this call; the caller samples different
                # (possibly shorter) parents next time.
                resp_text = getattr(getattr(e, 'response', None), 'text', '') or ''
                if 'ContextWindowExceeded' in resp_text or 'context window' in resp_text.lower():
                    print(f'Context window exceeded (prompt {len(content)} chars); skipping this call')
                    return ""
                if use_nvidia and self.share_api_key and not tried_share_key:
                    tried_share_key = True
                    headers["Authorization"] = f"Bearer {self.share_api_key}"
                    print(f'Exception {e},retry with share key')
                    continue
                # Rotate through fallback keys from the environment (comma
                # separated, set by the run scheduler) before falling back to
                # the 60s wait. Covers per-key rate limits during parallel runs.
                fallbacks = [k for k in os.getenv("NVIDIA_API_KEY_FALLBACKS", "").split(",")
                             if k and k != original_api_key]
                if use_nvidia and fallbacks:
                    idx = getattr(self, "_fallback_idx", 0)
                    if idx < len(fallbacks):
                        self._fallback_idx = idx + 1
                        headers["Authorization"] = f"Bearer {fallbacks[idx]}"
                        print(f'Exception {e},retry with fallback key #{idx+1}')
                        continue
                    self._fallback_idx = 0
                headers["Authorization"] = f"Bearer {original_api_key}"
                # 这里就是要一直retry，直到成功为止，不要改这里的代码
                print(f'Exception {e},retry in 60s, share key failed')
                time.sleep(60)
        usage = response_json['usage']
        output_tokens = int(usage['completion_tokens'])
        
        input_tokens = int(usage['prompt_tokens'])
        counted_output_tokens = output_tokens if output_tokens < 10000 else 0
        self._query_usage.tokens = (input_tokens, counted_output_tokens)
        with self._token_lock:
            self.input_tokens += input_tokens
            self.output_tokens += counted_output_tokens
        return output

    def _init_chat(self,model):
        if model == 'chatgpt':
            return self.gpt_chat
        elif model == 'llama':
            return self.llama_chat
        elif model == 'gemini':
            return self.gemini_chat
        elif model == 'deepseek':
            return self.deepseek_chat

    def _init_model(self,model):
        if model == 'chatgpt':
            return self._init_chatgpt()
        elif model == 'llama':
            return self._init_llama()
        elif model == 'gemini':
            return self._init_gemini()
        elif model == 'deepseek':
            return self._init_deepseek()

    def _init_deepseek(self):
        if OpenAI is None:
            raise ImportError("openai is required for the legacy deepseek client.")
        client = OpenAI(api_key="", base_url="https://api.deepseek.com")
        return client
    
    def deepseek_chat(self,content):
        response = self.model.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful chemist and biologist"},
                {"role": "user", "content": content},
            ],
            stream=False
        )
        return response.choices[0].message.content
    
    def _init_gemini(self):
        if genai is None:
            raise ImportError("google-generativeai is required for the legacy gemini client.")
        genai.configure(api_key="")
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model
    
    def gemini_chat(self,content):
        response = self.model.generate_content(content)
        print(response.text)
        return response.text

    def _init_llama(self):
        if OpenAI is None:
            raise ImportError("openai is required for the legacy llama client.")
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        return client
    
    def llama_chat(self,content):
        completion = self.model.chat.completions.create(
            model="NousResearch/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "user", "content": content}
            ]
        )
        return completion.choices[0].message.content
        


    def _init_chatgpt(self):
        if AzureOpenAI is None:
            raise ImportError("openai is required for the legacy Azure OpenAI client.")
        # Set the necessary variables
        resource_name = "ds-chatgpt4o-ai-swedencentral"#"gcrgpt4aoai2c" sfm-openai-sweden-central  ds-chatgpt4o-ai-swedencentral
        endpoint = f"https://{resource_name}.openai.azure.com/"
        api_version = "2024-02-15-preview"  # Replace with the appropriate API version

        azure_credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                # Exclude other credentials we are not interested in.
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            )
        )

        token_provider = get_bearer_token_provider(azure_credential,
            "https://cognitiveservices.azure.com/.default")
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )
        
        
        return client
    
    def gpt_chat(self, content):
        completion = self.model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who can propose novel and powerful molecules based on your domain knowledge."},
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )
        res = completion.choices[0].message.content
        return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test the LLM API client.")
    parser.add_argument(
        "--model",
        default=f"gcp/google/gemini-2.5-flash",
        help=(
            "Model spec. Use 'nvidia,<model>' for NVIDIA API. "
            "If no provider is given and the model is not a legacy name, NVIDIA is used."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="谜题1：1、2、4、6、7、8、9、10（打一成语），好好思考一下告诉我，只用回答这个成语不用解释",
        help="Prompt used for the test query.",
    )
    with open("../data.json", "r") as f:
        data = json.load(f)
    
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature.")
    args = parser.parse_args()
    args.prompt = data['messages'][0]['content']

    config = {"model": {"timeout": args.timeout}}
    if args.temperature is not None:
        config["model"]["temperature"] = args.temperature

    client = LLM(model=args.model, config=config)
    
    output = client.chat(args.prompt)

    print("\nOUTPUT:")
    print(output)
    print("\nTOKEN STATS:")
    print("input_tokens:", client.input_tokens)
    print("output_tokens:", client.output_tokens)
    
