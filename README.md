# Agent Kodu 🤖

Autonomiczny agent programowania, którym sterujesz z przeglądarki w telefonie.
Wpisujesz polecenie („zrób mi aplikację…”), agent sam planuje, tworzy pliki i foldery,
uruchamia polecenia, testuje i kończy pracę — **na serwerze**, więc możesz zamknąć przeglądarkę.

- Modele: **Google Gemini** i **NVIDIA NIM** (build.nvidia.com) — przez Twój klucz API
- Narzędzia agenta: drzewo plików, odczyt, zapis, edycja, przenoszenie, usuwanie, wyszukiwanie, terminal (bash), plan, zakończenie
- Panel mobilny: lista zadań, log na żywo, plan, przeglądarka plików, pobieranie ZIP, kolejne polecenia do tego samego projektu
- Odporność: automatyczne ponawianie przy limitach API (429/5xx), zapis po każdym kroku, **wznawianie po restarcie serwera**
- Opcjonalne powiadomienie push na telefon (ntfy.sh), gdy agent skończy
- Zero zależności — tylko Python 3.10+ (biblioteka standardowa)

## Szybki start (lokalnie / VPS)

```bash
APP_PASSWORD=twoje-haslo GEMINI_API_KEY=... NVIDIA_API_KEY=... python3 app.py
# otwórz http://IP-serwera:8080
```

Klucze możesz też wkleić później w panelu (⚙︎ Ustawienia).
Bez `APP_PASSWORD` hasło zostanie wygenerowane i wypisane w logach przy starcie.

### Docker

```bash
docker build -t agent-kodu .
docker run -d --restart unless-stopped -p 8080:8080 -v agent-data:/data \
  -e APP_PASSWORD=twoje-haslo -e GEMINI_API_KEY=... -e NVIDIA_API_KEY=... agent-kodu
```

Obraz zawiera Pythona, Node.js/npm, git i kompilator — agent może budować aplikacje webowe, skrypty, API itp.

## Wdrożenie na Render (bez komputera, z telefonu)

1. Wrzuć ten folder do nowego repozytorium na GitHubie.
2. Render → **New → Blueprint** → wybierz repo (użyje `render.yaml`).
3. Ustaw `APP_PASSWORD` (oraz opcjonalnie klucze API).
4. Otwórz adres `https://…onrender.com` w telefonie → „Dodaj do ekranu głównego”.

**Plan Starter (płatny, z dyskiem)** — `render.yaml` domyślnie: pliki projektów przetrwają restarty i redeploye.

**Plan Free** — usuń sekcję `disk` i zmień `plan: free`. Uwaga:
- darmowa instancja usypia po ~15 min bez ruchu; agent sam się pinguje (`RENDER_EXTERNAL_URL`), dopóki pracuje, więc zadanie dokończy się po zamknięciu przeglądarki,
- **brak trwałego dysku** — po restarcie/redeployu pliki znikają. Pobieraj wynik jako ZIP od razu po zakończeniu.

Inne hostingi z Dockerem (Railway, Fly.io, dowolny VPS) też działają — ważne, żeby proces działał stale i miał zapisywalny `/data`.

## Zmienne środowiskowe

| Zmienna | Opis |
|---|---|
| `APP_PASSWORD` | hasło do panelu (zalecane) |
| `GEMINI_API_KEY` / `NVIDIA_API_KEY` | klucze API (lub wpisz w panelu) |
| `DATA_DIR` | gdzie trzymać bazę i projekty (domyślnie `./data`) |
| `PORT` | port (domyślnie 8080) |
| `MAX_PARALLEL_JOBS` | ile zadań naraz (domyślnie 2) |
| `MAX_OUTPUT_TOKENS` | limit odpowiedzi modelu (domyślnie 16384, sam się zmniejsza, jeśli model ma mniej) |
| `MAX_CONTEXT_CHARS` | po przekroczeniu stare wyniki narzędzi są skracane (domyślnie 300000) |
| `CMD_TIMEOUT_MAX` | maks. czas jednego polecenia w terminalu, s (domyślnie 600) |
| `NTFY_TOPIC` | temat ntfy.sh do powiadomień push (lub w panelu) |
| `KEEPALIVE_URL` | własny publiczny adres do auto-pingu (na Renderze ustawiany automatycznie) |

## Modele

- Gemini: domyślnie `gemini-flash-latest` (alias na najnowszy Flash). Do trudnych projektów spróbuj `gemini-pro-latest`.
- NVIDIA: domyślnie `nvidia/nemotron-3-super-120b-a12b`; dobre do kodu są też np. modele Qwen Coder, Kimi, GLM.
- W formularzu przycisk ↻ pobiera aktualną listę modeli prosto z API. Model musi obsługiwać **tool calling**.

## Jak to działa

```
telefon (panel) ──HTTP──► serwer app.py ──► kolejka zadań ──► wątek agenta
                                                        │
                    Gemini / NVIDIA (API zgodne z OpenAI) ◄──► narzędzia: pliki + bash
                                                        │
                                    SQLite (log, historia) + data/workspaces/<id>/
```

Każde zadanie ma własny folder projektu. Po zakończeniu możesz dopisać kolejne polecenie
(np. „dodaj logowanie”) — agent kontynuuje w tym samym projekcie z pełną historią.
Gdy agent dojdzie do limitu kroków, zadanie jest „wstrzymane” — napisz „kontynuuj”.

## Bezpieczeństwo

Agent wykonuje polecenia powłoki na serwerze. Dlatego:
- zawsze ustaw mocne `APP_PASSWORD` i nie udostępniaj adresu,
- uruchamiaj w kontenerze/osobnym serwerze, nie na komputerze z ważnymi danymi,
- klucze API nie są przekazywane do poleceń uruchamianych przez agenta; operacje na plikach są ograniczone do folderu projektu
  (terminal działa w folderze projektu, ale technicznie ma dostęp do całego kontenera).
