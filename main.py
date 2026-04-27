import os
import time
import chromadb
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import logging
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import pandas as pd
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
import gc
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
import re
import socket
from langgraph.prebuilt import create_react_agent


# --- Configuration ---
MODEL_NAME = "C:\\Affan\\Affan\\__CODING__\\__PROJECTS__\\_______TEST______\\tinyllama_1_1b_chat"
EMBED_MODEL_NAME = "C:\\Affan\\Affan\\__CODING__\\__PROJECTS__\\__MINI PROJECT__\\bge_small_en_v1_5_local"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "my_rag_collection"
DOCS_DIR = "./uploaded_docs"
FEEDBACK_DB_PATH = "./feedback.db"
LORA_PATH = "./lora_finetuned"
PYTORCH_MODEL_PATH = "./pytorch_model"
OUTPUT_DIR = "output"
TEMP_DIR = "temp"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(LORA_PATH, exist_ok=True)
os.makedirs(PYTORCH_MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Load environment variables
load_dotenv()

# --- Flask App ---
app = Flask(__name__)
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:62020", "http://localhost:*"],  # Allow Flutter app origin
        "methods": ["GET", "POST", "OPTIONS"],  # Explicitly allow methods
        "allow_headers": ["Content-Type", "Authorization"]  # Allow specific headers
    }
})
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs.log'),
        logging.StreamHandler()
    ]
)

@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return jsonify({"status": "ok"}), 200

# Log all incoming requests with headers and full URL
@app.before_request
def log_request():
    logging.info(f"Incoming request: {request.method} {request.url} Headers: {request.headers} Payload: {request.get_json(silent=True) or request.form or request.data}")

# Custom 404 handler
@app.errorhandler(404)
def not_found(error):
    logging.error(f"404 error: {request.url}")
    return jsonify({"error": f"Endpoint {request.path} not found"}), 404

# --- SQLite Database for Feedback ---
def init_feedback_db():
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            message_id TEXT PRIMARY KEY,
            query TEXT,
            response TEXT,
            feedback TEXT,
            reward REAL
        )
    ''')
    conn.commit()
    conn.close()

# --- LLM Initialization ---
def initialize_llm():
    logging.info(f"Loading LLM from {MODEL_NAME}...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        llm = HuggingFaceLLM(
            model_name=MODEL_NAME,
            tokenizer_name=MODEL_NAME,
            context_window=768,
            max_new_tokens=128,
            generate_kwargs={"temperature": 0.1, "do_sample": True},
            device_map="auto",
            model_kwargs={"quantization_config": quantization_config},
            tokenizer_kwargs={}
        )
        logging.info("LLM loaded successfully!")
        return llm, None
    except Exception as e:
        logging.error(f"LLM load failed: {e}")
        return None, {"error": f"LLM load failed: {str(e)}"}

# --- Embedding Model ---
def initialize_embedding_model():
    logging.info("Loading embedding model...")
    try:
        embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logging.info("Embedding model loaded.")
        return embed_model, None
    except Exception as e:
        logging.error(f"Embedding model load failed: {e}")
        return None, {"error": f"Embedding model load failed: {str(e)}"}

# --- ChromaDB ---
def initialize_chromadb():
    try:
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return db.get_or_create_collection(COLLECTION_NAME), None
    except Exception as e:
        logging.error(f"ChromaDB initialization failed: {e}")
        return None, {"error": f"ChromaDB initialization failed: {str(e)}"}

# --- Global Settings ---
llm, llm_error = initialize_llm()
embed_model, embed_error = initialize_embedding_model()
chroma_collection, chroma_error = initialize_chromadb()
init_feedback_db()

if llm_error or embed_error or chroma_error:
    logging.error("Initialization failed")
    app.logger.error(f"Errors: {llm_error}, {embed_error}, {chroma_error}")
    raise Exception("Failed to initialize backend components")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
Settings.num_workers = 2

# --- RAG Index State ---
index = None
query_engine = None
index_ready = False

# --- Build/Reload RAG Index ---
def build_or_reload_index():
    global index, query_engine, index_ready
    index_ready = False
    query_engine = None
    try:
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        if not documents:
            return {"warning": "No valid documents found in uploaded_docs directory."}
        for doc in documents:
            logging.info(f"Loaded document: {doc.metadata.get('file_name', 'N/A')} with {len(doc.text)} characters")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        query_engine = index.as_query_engine(similarity_top_k=1)
        index_ready = True
        return {"success": "RAG index built successfully."}
    except Exception as e:
        logging.error(f"Indexing failed: {e}")
        return {"error": f"Indexing failed: {str(e)}"}

# --- Reward Model ---
def compute_reward(feedback, response_text):
    word_count = len(response_text.split())
    if word_count <= 50:
        return 1.0
    else:
        return -1.0

# --- Fine-Tuning with LoRA ---
class FineTuneModelManager:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, self.lora_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            max_new_tokens=128,
            temperature=0.1,
            return_full_text=False,
            batch_size=1,
        )
        trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        logging.info(f"Trainable parameters: {trainable_params}")

    def generate_response(self, query):
        try:
            prompt = f"<|user|>{query}<|end|><|assistant|> "
            response = self.pipeline(
                prompt,
                max_length=512,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )[0]['generated_text']
            cleaned_response = response.replace(prompt, "").strip()
            if not cleaned_response:
                logging.warning("Empty response generated for query: %s", query)
                return "I'm sorry, I couldn't generate a meaningful response. Please try rephrasing your query."
            logging.info("Generated response: %s", cleaned_response)
            return cleaned_response
        except Exception as e:
            logging.error(f"Pipeline generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def fine_tune(self, queries, responses, rewards):
        start_time = time.time()
        try:
            data = [{"query": q, "response": r, "reward": rw} for q, r, rw in zip(queries, responses, rewards) if rw != 0]
            if not data:
                logging.info("No valid feedback data for fine-tuning.")
                return {"error": "No valid feedback data available for fine-tuning."}
            formatted_data = [
                {"text": f"<|user|>{d['query']}<|end|><|assistant|>{d['response']}<|end|>"} 
                for d in data
            ]
            dataset = Dataset.from_list(formatted_data)
            sft_config = SFTConfig(
                output_dir=LORA_PATH,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=2e-5,
                max_steps=10,
                logging_steps=1,
                optim="adamw_torch",
                dataset_text_field="text",
                max_seq_length=512,
                label_names=None,
            )
            self.model.train()
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=dataset,
                args=sft_config,
            )
            trainer.train()
            self.model.save_pretrained(LORA_PATH)
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(PYTORCH_MODEL_PATH)
            self.tokenizer.save_pretrained(PYTORCH_MODEL_PATH)
            logging.info(f"SFT fine-tuning completed in {time.time() - start_time:.2f} seconds")
            global llm
            llm, llm_error = initialize_llm()
            if llm_error:
                logging.error(f"Failed to reload LLM: {llm_error}")
                return {"error": f"Failed to reload LLM: {llm_error}"}
            Settings.llm = llm
            if index_ready:
                build_or_reload_index()
            return {"success": "Fine-tuning completed successfully."}
        except Exception as e:
            logging.error(f"SFT fine-tuning failed: {e}")
            return {"error": f"SFT fine-tuning failed: {str(e)}"}

    def load_finetuned_model(self):
        if os.path.exists(LORA_PATH):
            self.model = PeftModel.from_pretrained(self.model, LORA_PATH)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                max_new_tokens=128,
                temperature=0.1,
                return_full_text=False,
                batch_size=1,
            )
            logging.info("Loaded fine-tuned LoRA weights.")

# Initialize fine-tuning model manager
fine_tune_manager = FineTuneModelManager()

# --- Image Generation ---
def generate_image(prompt, model_path="C:\\Affan\\Affan\\__CODING__\\__PROJECTS__\\flutter_scribble-main\\scribble_to_image\\backend\\stable-diffusion-v1-5"):
    try:
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype
        ).to(device)
        image = pipe(prompt).images[0]
        image_filename = f"generated_{hash(prompt)}_{int(time.time())}.png"
        image_path = os.path.join(OUTPUT_DIR, image_filename)
        image.save(image_path)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        return {"image_path": image_filename, "image_base64": img_str}
    except Exception as e:
        logging.error(f"Error in generate_image: {e}")
        return {"error": str(e)}

# --- Scribble to Image ---
def load_scribble_image(image_data):
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = image.resize((512, 512))
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(image_rgb)

def generate_from_scribble(scribble_data, prompt, base_model_path="C:\\Affan\\Affan\\__CODING__\\__PROJECTS__\\flutter_scribble-main\\scribble_to_image\\backend\\stable-diffusion-v1-5"):
    try:
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        controlnet = ControlNetModel.from_pretrained(
            "C:\\Affan\\Affan\\__CODING__\\__PROJECTS__\\flutter_scribble-main\\scribble_to_image\\backend\\sd-controlnet-scribble",
            torch_dtype=torch.float16
        ).to(device)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        scribble_image = load_scribble_image(scribble_data)
        image = pipe(prompt=prompt, image=scribble_image, num_inference_steps=30).images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        return {"image_base64": img_str}
    except Exception as e:
        logging.error(f"Error in generate_from_scribble: {e}")
        return {"error": str(e)}

# --- WhatsApp Agent ---
async def initialize_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logging.warning("GROQ_API_KEY not found. WhatsApp agent disabled.")
        return None
    try:
        client = MultiServerMCPClient(
            {
                "whatsapp": {
                    "command": "C:\\Users\\Dell\\.local\\bin\\uv.exe",
                    "args": [
                        "--directory",
                        "C:\\Affan\\Affan\\__CODING__\\__PROJECTS__\\whatsapp-mcp-main\\whatsapp-mcp-main\\whatsapp-mcp-server",
                        "run",
                        "main.py"
                    ],
                    "transport": "stdio",
                }
            }
        )
        os.environ["GROQ_API_KEY"] = groq_api_key
        tools = await client.get_tools()
        model = ChatGroq(model="llama3-8b-8192")
        return create_react_agent(model, tools)
    except Exception as e:
        logging.error(f"Failed to initialize WhatsApp agent: {e}")
        return None

# Initialize WhatsApp agent lazily
agent = None

# --- Game Generation ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    game_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    logging.warning("GOOGLE_API_KEY not found. Game generation disabled.")
    game_model = None

# --- API Endpoints ---
@app.route('/api/upload', methods=['POST'])
def upload_files():
    logging.info("Handling /api/upload request")
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    files = request.files.getlist('files')
    uploaded_files = []
    for file in files:
        if file and file.filename.endswith(('.txt', '.md', '.pdf')):
            file_path = os.path.join(DOCS_DIR, file.filename)
            try:
                file.save(file_path)
                uploaded_files.append(file.filename)
            except Exception as e:
                logging.error(f"Failed to save file {file.filename}: {e}")
                return jsonify({"error": f"Failed to save file {file.filename}: {str(e)}"}), 500
    if uploaded_files:
        return jsonify({"success": f"Uploaded {len(uploaded_files)} files: {', '.join(uploaded_files)}"})
    return jsonify({"error": "No valid files uploaded"}), 400

@app.route('/api/build_index', methods=['POST'])
def build_index():
    logging.info("Handling /api/build_index request")
    result = build_or_reload_index()
    return jsonify(result)

@app.route('/api/query', methods=['POST'])
def query():
    logging.info("Handling /api/query request")
    start_time = time.time()
    data = request.get_json()
    user_query = data.get('query', '').strip()
    if not user_query:
        return jsonify({"error": "Please provide a query"}), 400
    try:
        if index_ready and query_engine:
            response = query_engine.query(user_query)
            sources = [
                {
                    "text": node.text,
                    "score": node.score,
                    "file_name": node.metadata.get('file_name', 'N/A')
                } for node in response.source_nodes
            ]
            logging.info(f"RAG query took {time.time() - start_time:.2f} seconds")
            return jsonify({
                "response": response.response,
                "sources": sources,
                "mode": "RAG",
                "id": str(hash(user_query + response.response))
            })
        else:
            response = fine_tune_manager.generate_response(user_query)
            logging.info(f"Chatbot query took {time.time() - start_time:.2f} seconds")
            return jsonify({
                "response": response,
                "sources": [],
                "mode": "Chatbot",
                "id": str(hash(user_query + response))
            })
    except Exception as e:
        logging.error(f"Error during response: {e}")
        return jsonify({"error": f"Error during response: {str(e)}"}), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    logging.info("Handling /api/feedback request")
    data = request.get_json()
    message_id = data.get('message_id')
    query = data.get('query')
    response = data.get('response')
    feedback_type = data.get('feedback')
    if not all([message_id, query, response, feedback_type]):
        return jsonify({"error": "Missing required fields: message_id, query, response, feedback"}), 400
    try:
        reward = compute_reward(feedback_type, response)
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO feedback (message_id, query, response, feedback, reward) VALUES (?, ?, ?, ?, ?)',
            (message_id, query, response, feedback_type, reward)
        )
        conn.commit()
        conn.close()
        return jsonify({"success": "Feedback recorded successfully"})
    except Exception as e:
        logging.error(f"Error recording feedback: {e}")
        return jsonify({"error": f"Error recording feedback: {str(e)}"}), 500

@app.route('/api/finetune', methods=['POST'])
def finetune():
    logging.info("Handling /api/finetune request")
    start_time = time.time()
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        df = pd.read_sql_query("SELECT query, response, reward FROM feedback WHERE reward != 0", conn)
        conn.close()
        if len(df) < 10:
            return jsonify({"error": f"Insufficient feedback entries: {len(df)} found, minimum 10 required."}), 400
        queries = df['query'].tolist()
        responses = df['response'].tolist()
        rewards = df['reward'].tolist()
        result = fine_tune_manager.fine_tune(queries, responses, rewards)
        logging.info(f"Fine-tuning completed in {time.time() - start_time:.2f} seconds")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        return jsonify({"error": f"Error during fine-tuning: {str(e)}"}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    logging.info("Handling /generate request")
    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    result = generate_image(prompt)
    if "error" in result:
        return jsonify({"error": result["error"]}), 500
    return jsonify({
        "image_url": f"/output/{result['image_path']}",
        "image_base64": result["image_base64"]
    })

@app.route('/output/<filename>', methods=['GET'])
def serve_image(filename):
    logging.info(f"Handling /output/{filename} request")
    try:
        return send_from_directory(OUTPUT_DIR, filename)
    except Exception as e:
        logging.error(f"Error serving image {filename}: {e}")
        return jsonify({"error": f"Image {filename} not found"}), 404

@app.route('/api/generate_scribble', methods=['POST'])
def generate_scribble():
    logging.info("Handling /generate_scribble request")
    data = request.get_json()
    scribble_base64 = data.get('scribble_image')
    prompt = data.get('prompt', '')
    if not scribble_base64 or not prompt:
        return jsonify({"error": "Scribble image and prompt are required"}), 400
    result = generate_from_scribble(scribble_base64, prompt)
    if "error" in result:
        return jsonify({"error": result["error"]}), 500
    return jsonify(result)

@app.route('/api/send_message', methods=['POST'])
def send_message():
    logging.info("Handling /send_message request")
    global agent
    if not agent:
        agent = asyncio.run(initialize_agent())
        if not agent:
            return jsonify({"error": "WhatsApp agent not initialized. Please ensure GROQ_API_KEY is set."}), 500
    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    try:
        prompt_lower = prompt.lower()
        if "mom" in prompt_lower:
            content = f"Send a {prompt} message to 919966960075"
        elif "imran" in prompt_lower:
            content = f"Send a {prompt} message to 919701918796"
        else:
            content = f"{prompt}"
        msg_response = asyncio.run(agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }
        ))
        return jsonify({"response": msg_response['messages'][-1].content})
    except Exception as e:
        logging.error(f"Error in send_message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate_game', methods=['POST'])
def generate_game():
    logging.info("Handling /generate_game request")
    try:
        if game_model is None:
            return jsonify({'error': 'Game generation is disabled. Set GOOGLE_API_KEY to enable it.'}), 503
        data = request.get_json()
        prompt_input = data.get('prompt', '').strip()
        if not prompt_input:
            return jsonify({'error': 'Enter a game description.'}), 400
        prompt = f"""
You are a Python developer.
Generate a Python game with the following details:
Game: {prompt_input}
Instructions:
1. Output only clean, executable Python code. No comments, no explanations, no markdown.
2. After the code, list pip-installable packages (one per line).
3. Then, provide clear steps to run the game.
Respond in the following structure:
<code starts here>
... code ...
<code ends here>
<dependencies start>
... packages ...
<dependencies end>
<steps start>
... steps ...
<steps end>
"""
        response = game_model.generate_content(prompt)
        full_output = response.text.strip()
        def extract_between_tags(tag_start, tag_end, text):
            match = re.search(rf"{re.escape(tag_start)}(.*?){re.escape(tag_end)}", text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else None
        code = extract_between_tags("<code starts here>", "<code ends here>", full_output)
        deps_raw = extract_between_tags("<dependencies start>", "<dependencies end>", full_output)
        steps = extract_between_tags("<steps start>", "<steps end>", full_output)
        if not code:
            code_match = re.search(r"```(?:python)?\s*(.*?)```", full_output, re.DOTALL)
            code = code_match.group(1).strip() if code_match else None
        if not code:
            return jsonify({'error': 'Failed to extract code. Please try rephrasing your game prompt.'}), 500
        code = re.sub(r"^```(?:python)?\s*", "", code).strip()
        code = re.sub(r"```$", "", code).strip()
        dependencies = []
        if deps_raw:
            dependencies = [line.strip() for line in deps_raw.splitlines() if line.strip()]
        os.makedirs("output", exist_ok=True)
        with open("output/game.py", "w", encoding="utf-8") as f:
            f.write(code)
        with open("output/requirements.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(dependencies) if dependencies else "# No external dependencies")
        with open("output/README.md", "w", encoding="utf-8") as f:
            f.write(f"# How to Run the Game\n\n{steps if steps else 'Instructions not provided.'}")
        batch_file = os.path.join("zaki", "1.bat")
        if os.path.exists(batch_file):
            print("⚙️ Launching run_game.bat independently...")
            os.startfile(batch_file)
        else:
            return jsonify({'error': 'run_game.bat not found in output directory.'}), 500    
        return jsonify({
            'code': code,
            'dependencies': dependencies,
            'steps': steps if steps else 'Instructions not provided.'
        })
    except Exception as e:
        logging.error(f"Error in generate_game: {e}")
        return jsonify({'error': str(e)}), 500

# --- Debug Route ---
@app.route('/debug/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':  # Exclude static route
            routes.append({
                "endpoint": rule.endpoint,
                "methods": list(rule.methods),
                "path": str(rule)
            })
    logging.info(f"Listing routes: {routes}")
    return jsonify({"routes": routes})

# --- Health Check ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Server is running"})

if __name__ == '__main__':
    logging.info("Starting app.py")
    try:
        # Check if port is available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('0.0.0.0', 5000))
        if result == 0:
            logging.error("Port 5000 is already in use.")
            raise Exception("Port 5000 is already in use.")
        sock.close()
        
        # Log all registered routes
        logging.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            if rule.endpoint != 'static':
                logging.info(f"Endpoint: {rule.endpoint}, Path: {rule}, Methods: {rule.methods}")
        
        # Start server
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        # For production, uncomment the following and comment out app.run:
        # from waitress import serve
        # serve(app, host='0.0.0.0', port=5000, threads=8)
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        raise