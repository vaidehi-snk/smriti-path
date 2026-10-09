# SmritiPath

Turn a spoken or written memory of a place into a walking route you take together.
Speech-to-text and stop-finding run locally with open-source models. Nothing is sent to any AI service.

## How it works
1. Add a recording (faster-whisper) or paste a memory as text.
2. A local model (gemma3:4b via Ollama) finds each place mentioned, with a quote from the speaker.
   Code checks every quote really appears in the transcript.
3. Pin the stops on a map (Leaflet + OpenStreetMap).
4. Download the Walk Pack: one offline page with a sketch map, "then" stories,
   questions to ask, and blank "Now" lines to fill in by pen on the walk.

## Run it
    pip install -r requirements.txt
    ollama pull gemma3:4b
    uvicorn server:app
Open http://127.0.0.1:8000

Audio and transcripts are git-ignored and never leave your machine. Map tiles and place search need internet while building; the Walk Pack itself works offline.