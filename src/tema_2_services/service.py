import json
import os
import hashlib
import logging

from dotenv import load_dotenv
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "data")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "data_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss.index.meta")
USE_MODEL_URL = os.environ.get(
    "USE_MODEL_URL",
    "https://tfhub.dev/google/universal-sentence-encoder/4",
)

WEB_URLS = [u for u in os.environ.get("WEB_URLS", "").split(";") if u]
if not WEB_URLS:
    raise ValueError("Seteaza WEB_URLS in .env (URL-uri separate prin ';')")

class RAGAssistant:
    """Asistent cu RAG din surse web si un LLM pentru raspunsuri."""

    def __init__(self) -> None:
        """Initializeaza clientul LLM, embedderul si prompturile."""
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Seteaza GROQ_API_KEY in variabilele de mediu.")

        self.groq_base_url = os.environ.get("GROQ_BASE_URL")
        if not self.groq_base_url:
            raise ValueError("Seteaza GROQ_BASE_URL in variabilele de mediu.")

        self.client = OpenAI(
            api_key=self.groq_api_key,
            base_url=self.groq_base_url,
        )

        os.makedirs(DATA_DIR, exist_ok=True)
        self.embedder = None
        self._relevance = None

        self.system_prompt = (
            "Esti un asistent specializat in analiza si interpretarea sesiunilor "
            "dispozitivelor avansate de biofeedback. "
            "Raspunzi doar la intrebari relevante despre semnale fiziologice "
            "(HRV, EEG, GSR, temperatura corporala, respiratie), stari psihologice "
            "si antrenament mental, pe baza contextului extras din sursele furnizate. "
            "Ofera raspunsuri clare, concise si bine structurate. "
            "Daca informatia lipseste din context, spune explicit acest lucru si nu "
            "inventa detalii. "
            "Cand este util, foloseste bullet points sau pasi numerotati. "
            "Pastreaza raspunsul in limba romana."
        )

    @property
    def relevance(self):
        if self._relevance is None:
            self._relevance = self._embed_texts(
                "Aceasta este o intrebare relevanta despre analiza si interpretarea "
                "sesiunilor, dispozitivelor de biofeedback, inclusiv semnale "
                "fiziologice, HRV, coerenta cardiaca, undele cerebrale EEG, raspuns "
                "galvanic al pielii, stres, relaxare si antrenament mental.",
            )[0]
        return self._relevance

    def _load_documents_from_web(self) -> list[str]:
        """Incarca si chunked documente de pe site-uri prin WebBaseLoader."""
        if os.path.exists(CHUNKS_JSON_PATH):
            try:
                with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and cached:
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        all_chunks = []
        for url in WEB_URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                for doc in docs:
                    chunks = self._chunk_text(doc.page_content)
                    all_chunks.extend(chunks)
            except Exception as e:
                logger.warning("Nu am putut incarca %s: %s", url, e)
                continue

        if all_chunks:
            with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)

        return all_chunks

    def _send_prompt_to_llm(
        self,
        user_input: str,
        context: str,
    ) -> str:
        """Trimite promptul catre LLM si returneaza raspunsul."""
        system_msg = self.system_prompt

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    f"Context relevant despre biofeedback si semnale fiziologice:\n"
                    f"{context}\n\n"
                    f"Intrebarea utilizatorului:\n{user_input}\n\n"
                    "Raspunde in limba romana, clar si structurat. "
                    "Bazeaza-te in primul rand pe contextul furnizat. "
                    "Daca contextul nu contine informatia necesara, precizeaza "
                    "explicit acest lucru si nu genera detalii inventate."
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-20b",
                timeout=30,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception:
            return (
                "Asistent: Nu pot ajunge la modelul de limbaj acum. "
                "Te rog incearca din nou in cateva momente."
            )

    def _embed_texts(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Genereaza embeddings folosind Universal Sentence Encoder."""
        if isinstance(texts, str):
            texts = [texts]
        if self.embedder is None:
            self.embedder = hub.load(USE_MODEL_URL)
        if callable(self.embedder):
            embeddings = self.embedder(texts)
        else:
            infer = self.embedder.signatures.get("default")
            if infer is None:
                raise ValueError("Model USE nu expune semnatura 'default'.")
            outputs = infer(tf.constant(texts))
            embeddings = outputs.get("default")
            if embeddings is None:
                raise ValueError("Model USE nu a returnat cheia 'default'.")
        return np.asarray(embeddings, dtype="float32")

    def _chunk_text(self, text: str) -> list[str]:
        """Imparte textul in bucati cu RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
        )
        chunks = splitter.split_text(text or "")
        return [c for c in chunks if c.strip()] or []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculeaza similaritatea cosine intre doi vectori."""
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_faiss_index_from_chunks(self, chunks: list[str]) -> faiss.IndexFlatIP:
        """Construieste index FAISS din chunks text si il salveaza pe disc."""
        if not chunks:
            raise ValueError("Lista de chunks este goala.")

        embeddings = self._embed_texts(chunks).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            f.write(self._compute_chunks_hash(chunks))
        return index

    def _compute_chunks_hash(self, chunks: list[str]) -> str:
        """Hash determinist pentru lista de chunks si model."""
        payload = json.dumps(
            {
                "model": USE_MODEL_URL,
                "chunks": chunks,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_index_hash(self) -> str | None:
        """Incarca hash-ul asociat indexului FAISS."""
        if not os.path.exists(FAISS_META_PATH):
            return None
        try:
            with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _retrieve_relevant_chunks(
        self, chunks: list[str], user_query: str, k: int = 5
    ) -> list[str]:
        """Rankeaza chunks folosind FAISS si returneaza top-k relevante."""
        if not chunks:
            return []

        current_hash = self._compute_chunks_hash(chunks)
        stored_hash = self._load_index_hash()

        query_embedding = self._embed_texts(user_query).astype("float32")

        index = None
        if os.path.exists(FAISS_INDEX_PATH) and stored_hash == current_hash:
            try:
                index = faiss.read_index(FAISS_INDEX_PATH)
                if (
                    index.ntotal != len(chunks)
                    or index.d != query_embedding.shape[1]
                ):
                    index = None
            except Exception:
                index = None

        if index is None:
            index = self._build_faiss_index_from_chunks(chunks)

        faiss.normalize_L2(query_embedding)

        k = min(k, len(chunks))
        if k == 0:
            return []

        _, indices = index.search(query_embedding, k=k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]

    def calculate_similarity(self, text: str) -> float:
        """Returneaza similaritatea cu propozitia de referinta biofeedback."""
        embedding = self._embed_texts(text.strip())[0]
        return self._cosine_similarity(embedding, self.relevance)

    def is_relevant(self, user_input: str) -> bool:
        """Verifica daca intrarea utilizatorului este despre biofeedback."""
        return self.calculate_similarity(user_input) >= 0.55

    def assistant_response(self, user_message: str) -> str:
        """Directioneaza mesajul utilizatorului catre calea potrivita."""
        if not user_message or not user_message.strip():
            return (
                "Te rog scrie o intrebare despre biofeedback. "
                "Exemplu: Ce este HRV si cum il interpretez intr-o sesiune "
                "de biofeedback?"
            )

        if not self.is_relevant(user_message):
            return (
                "Pot raspunde doar la intrebari despre analiza si interpretarea "
                "sesiunilor de biofeedback. De exemplu, poti intreba: Cum "
                "interpretez undele EEG inregistrate cu un dispozitiv de biofeedback?"
            )

        chunks = self._load_documents_from_web()
        if not chunks:
            return (
                "Nu am putut incarca sursele web configurate pentru biofeedback. "
                "Verifica variabila WEB_URLS din fisierul .env."
            )

        relevant_chunks = self._retrieve_relevant_chunks(chunks, user_message)
        context = "\n\n".join(relevant_chunks)
        return self._send_prompt_to_llm(user_message, context)

if __name__ == "__main__":
    assistant = RAGAssistant()
    print(assistant.assistant_response("Ce inseamna coerenta cardiaca si cum o masoara un dispozitiv de biofeedback?"))
    print(assistant.assistant_response("Cum interpretez o sesiune EEG inregistrata cu un dispozitiv biofeedback?"))
    print(assistant.assistant_response("Care este capitala Frantei?"))
    print(assistant.assistant_response("Cum fac un endpoint GET in FastAPI?"))
