

# Tema 2 - Construiți un asistent care să răspundă la informații relevante doar pentru ideea dumneavoastră.

# Cerinte

- Fork din github la repository-ul [https://github.com/dragosbajenaru1001/Teme\\_pentru\\_acasa](https://github.com/dragosbajenaru1001/Teme_pentru_acasa)
- Rezolvați cerințele To-Do din fisierul

[https://github.com/dragosbajenaru1001/Teme\\_pentru\\_acasa/blob/main/src/tema\\_2\\_services/service.py](https://github.com/dragosbajenaru1001/Teme_pentru_acasa/blob/main/src/tema_2_services/service.py)

---

## Instructiuni

- ### 1. Instalati Python 3.10.11 (o singura data)

<https://www.python.org/downloads/release/python-31011/>

- ### 2. Creati si activati virtualenv

```
```powershell  
python -m venv venv  
.venv\Scripts\activate  
pip install -r requirements.txt
```

## Configurati variabilele de mediu

Am obtinut un API key gratuit Groq la: <https://console.groq.com/>

Am creat un fisier .env in radacina repository-ului:

```
GROQ_API_KEY=your_groq_api_key_here  
GROQ_BASE_URL=https://api.groq.com/openai/v1  
DATA_DIR=data  
USE_MODEL_URL=https://tfhub.dev/google/universal-sentence-encoder/4
```

In variabila WEB\_URLS am pus paginile web pe care se va antrena modelul (separate prin ;):

```
WEB_URLS=https://www.heartmath.org/research/science-of-the-heart;/https://en.wikipedia.org/wiki/Biofeedback;https://en.wikipedia.org/wiki/Heart_rate_variability;https://en.wikipedia.org/wiki/Electroencephalography;https://en.wikipedia.org/wiki/Galvanic_skin_response;https://en.wikipedia.org/wiki/Neurofeedback;https://www.mayoclinic.org/tests-procedures/biofeedback/about/pac-20384664;https://my.clevelandclinic.org/health/treatments/11945-biofeedback;https://brainflow.readthedocs.io/en/stable/;https://mne.tools/stable/index.html;https://neuroopsychology.github.io/NeuroKit/;https://docs.openbci.com/;https://physionet.org/
```

```
TF_ENABLE_ONEDNN_OPTS=0
```

**Nota:** Lista completa de URL-uri biofeedback se gaseste in fisierul .env

### Rulare

```
python service.py
```

**Nota:** Prima rulare descarca si indexeaza toate URL-urile din WEB\_URLS — poate dura cateva minute. Rulari ulterioare folosesc cache-ul din data/.

### Resetare cache (dupa modificarea WEB\_URLS)

```
Remove-Item -Force .\data\data_chunks.json, .\data\faiss.index, .\data\faiss.index.meta -ErrorAction SilentlyContinue
```

## Rezolvare

## Agent AI Biofeedback — service.py

Un agent RAG (Retrieval-Augmented Generation) specializat in analiza si interpretarea sesiunilor dispozitivelor avansate de biofeedback. Raspunde doar la intrebari relevante despre semnale fiziologice (HRV, EEG, GSR, EMG, temperatura corporala), stari psihologice si antrenament mental, ignorand intrebarile din afara domeniului.

## Cum functioneaza

Intrebare utilizator

![](7055f51feb10ea4ea48b27c36f085286_img.jpg)

|  
▼

Verificare  
relevanta  
biofeedback

Embeddings (Universal Sentence Encoder)  
Similaritate cosine  $\geq 0.55$  cu propozitia  
de referinta domeniului

| irelevant → raspuns de respingere  
| relevant  
▼

Incarcare date din surse web  
  
Web scraping (WebBaseLoader + BeautifulSoup)  
100+ URL-uri din domeniu biofeedback:  
- HeartMath, Mayo Clinic, Cleveland Clinic  
- Wikipedia: HRV, EEG, GSR, EMG, Neurofeedback  
- BrainFlow, MNE, NeuroKit, OpenBCI docs  
- PhysioNet, PubMed, arXiv papers  
- Emotiv, NeuroSky, Neurosity device docs  
Chunked (300 chars, overlap 20) si cached

![](7a1dee155822446f7828dcb055c465c3_img.jpg)

|  
▼

Retrieval semantic FAISS (top-5)  
  
IndexFlatIP cu embeddings USE  
Hash determinist pentru invalidare cache  
Rebuildit automat la schimbari WEB\_URLS

![](f89631cc38aa971e8d15cbffe28f1183_img.jpg)

|  
▼

LLM (Groq)  
openai/gpt-oss-20b

System prompt + context + intrebare → raspuns  
Raspunde in romana, clar si structurat

## Structura proiectului

```
Tema2/
├── service.py          	# Agentul principal RAG Biofeedback
├── requirements.txt    	# Dependinte Python
├── .env              		# Variabile de mediu (nu se comite in git)
├── README.md
└── data/             		# Date si cache (generate automat)
    ├── data_chunks.json    # Cache chunks web (generat automat)
    ├── faiss.index       	# Index FAISS (generat automat)
    └── faiss.index.meta  	# Hash pentru invalidare cache (generat automat)
```

### Variabile de mediu

| Variabila             | Descriere                                           | Obligatorie                                                                                |
|-----------------------|-----------------------------------------------------|--------------------------------------------------------------------------------------------|
| GROQ_API_KEY          | API key pentru Groq LLM                             | Da                                                                                         |
| GROQ_BASE_URL         | Endpoint Groq OpenAI-compatible                     | Da (default: <a href="https://api.groq.com/openai/v1">https://api.groq.com/openai/v1</a> ) |
| DATA_DIR              | Director pentru cache FAISS si chunks               | Nu (default: data)                                                                         |
| WEB_URLS              | URL-uri separate prin ; pentru scraping biofeedback | Da                                                                                         |
| USE_MODEL_URL         | URL model Universal Sentence Encoder                | Nu (are default TFHub v4)                                                                  |
| TF_ENABLE_ONEDNN_OPTS | Dezactiveaza warning-uri TensorFlow                 | Nu                                                                                         |

## Surse de cunoastere indexate

Agentul indexeaza peste 100 de URL-uri din domenii relevante biofeedback:

| Categorie                       | Surse                                                      |
|---------------------------------|------------------------------------------------------------|
| Fundamente biofeedback          | HeartMath, Mayo Clinic, Cleveland Clinic, AAPB, BCIA       |
| Semnale fiziologice (Wikipedia) | HRV, EEG, GSR, EMG, Neurofeedback, Electrodermal Activity  |
| Librarii si tooling Python      | BrainFlow, MNE-Python, NeuroKit2, BioSPPy, PyWavelets      |
| Dispozitive hardware            | OpenBCI (Cyton/Ganglion), Emotiv EPOC, NeuroSky, Neurosity |
| Seturi de date stiintifice      | PhysioNet, OpenNeuro, DEAP, AMIGOS, SEED, MAHNOB           |
| Cercetare academica             | PubMed (HRV+stress, EEG+ML, GSR+ML), arXiv preprints       |
| Standarde si formate            | BIDS, EDF/EDF+, HL7 FHIR, SDTM                             |

## Exemple de interactiune

Intrebare relevanta: "Ce reprezinta coerenta cardiaca in biofeedback?"

Raspuns: Explicatie detaliata HRV si coerenta cardiaca din contextul HeartMath + Wikipedia + PubMed

Intrebare relevanta: "Cum se interpreteaza undele theta in EEG?"

Raspuns: Descriere unde cerebrale, asocieri cognitive, aplicatii neurofeedback din contextul MNE + eeginfo

Intrebare relevanta: "Ce este GSR si cum se masoara?"

Raspuns: Explicatie raspuns galvanic al pielii, metodologie de masurare, aplicatii clinice

Intrebare irelevanta: "Care este reteta de sarmale?"

Raspuns: "Intrebarea ta nu pare a fi despre analiza sesiunilor de biofeedback. Te rog sa adresezi intrebari legate de semnale fiziologice, HRV, EEG, GSR sau antrenament mental."

## Componente tehnice implementate

| Componenta                  | Implementare                          | Detalii                             |
|-----------------------------|---------------------------------------|-------------------------------------|
| <b>Embedding model</b>      | Universal Sentence Encoder v4 (TFHub) | Vectori 512-dimensionali            |
| <b>Verificare relevanta</b> | Similaritate cosine $\geq 0.55$       | Propozitie de referinta biofeedback |
| <b>Web scraping</b>         | LangChain WebBaseLoader               | 100+ URL-uri domeniu biofeedback    |
| <b>Chunking text</b>        | RecursiveCharacterTextSplitter        | chunk_size=300, overlap=20          |
| <b>Index vectorial</b>      | FAISS IndexFlatIP                     | Cache pe disc cu hash SHA-256       |
| <b>Retrieval</b>            | Cautare semantica FAISS top-5         | Normalizare L2 + inner product      |
| <b>LLM</b>                  | Groq API (OpenAI-compatible)          | Model openai/gpt-oss-20b            |
| <b>Sistem prompt</b>        | Specializat biofeedback               | Raspuns in romana, fara halucinatii |
| <b>Cache invalidare</b>     | Hash SHA-256 (URL-uri + model)        | Rebuild automat la schimbari        |

### System Prompt utilizat

Esti un asistent specializat in analiza si interpretarea sesiunilor dispozitivelor avansate de biofeedback. Raspunzi doar la intrebari relevante despre semnale fiziologice (HRV, EEG, GSR, temperatura corporala, respiratie), stari psihologice si antrenament mental, pe baza contextului extras din sursele furnizate. Oferi raspunsuri clare, concise si bine structurate. Daca informatia lipseste din context, spune explicit acest lucru si nu inventa detalii. Cand este util, foloseste bullet points sau pasi numerotati. Pastreaza raspunsul in limba romana.

## Propozitie de referinta relevanta (pentru filtrare domeniu)

Aceasta este o intrebare relevanta despre analiza si interpretarea sesiunilor, dispozitivelor de biofeedback, inclusiv semnale fiziologice, HRV, coerenta cardiaca, undele cerebrale EEG, raspuns galvanic al pielii, stres, relaxare si antrenament mental.
