import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import time
import json
import re
import html
import logging
import traceback
from datetime import datetime
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool

# ------------------------------------------------------------------
# Configuração via variáveis de ambiente (Railway)
# ------------------------------------------------------------------
API_KEY = os.environ["GEMINI_API_KEY"]
os.environ["GEMINI_API_KEY"] = API_KEY

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# Gmail OAuth — token.json salvo como variável de ambiente (JSON string)
GMAIL_TOKEN_JSON = os.environ["GMAIL_TOKEN_JSON"]
GMAIL_CREDENTIALS_JSON = os.environ.get("GMAIL_CREDENTIALS_JSON", "")

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("openclaw-bot")

# ------------------------------------------------------------------
# Telegram
# ------------------------------------------------------------------
def md_para_telegram(texto: str) -> str:
    texto = html.escape(texto)
    saida = []
    for linha in texto.split("\n"):
        s = linha.strip()
        if not s:
            saida.append("")
            continue
        if re.fullmatch(r"[-_*]{3,}", s):
            saida.append("──────────")
            continue
        m = re.match(r"^#{1,6}\s*(.*)", s)
        if m:
            saida.append(f"\n<b>{m.group(1).replace('**', '')}</b>")
            continue
        m = re.match(r"^(\s*)[*\-]\s+(.*)", linha)
        if m:
            recuo = "    ◦ " if len(m.group(1)) >= 2 else "• "
            linha = recuo + m.group(2)
        linha = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", linha)
        linha = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", linha)
        saida.append(linha)
    return "\n".join(saida).strip()

def enviar_telegram(texto: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    html_texto = md_para_telegram(texto)

    blocos, atual = [], ""
    for linha in html_texto.split("\n"):
        if len(atual) + len(linha) + 1 > 3800:
            blocos.append(atual)
            atual = ""
        atual += linha + "\n"
    if atual.strip():
        blocos.append(atual)

    for bloco in blocos:
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": bloco, "parse_mode": "HTML"},
            timeout=30,
        )
        if r.status_code == 400:
            limpo = re.sub(r"</?[bi]>", "", html.unescape(bloco))
            r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": limpo}, timeout=30)
        r.raise_for_status()

# ------------------------------------------------------------------
# Gmail — autenticação via token em variável de ambiente
# ------------------------------------------------------------------
def get_gmail_service():
    token_data = json.loads(GMAIL_TOKEN_JSON)
    creds = Credentials.from_authorized_user_info(token_data, SCOPES)

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            log.info("Token Gmail expirado, renovando...")
            creds.refresh(Request())
            # Atualiza a variável em memória (o Railway mantém a env original)
            log.info("Token Gmail renovado com sucesso.")
        else:
            raise RuntimeError(
                "Token Gmail inválido e sem refresh_token. "
                "Rode o bot localmente uma vez para re-autenticar."
            )

    return build('gmail', 'v1', credentials=creds)

# ------------------------------------------------------------------
# Ferramenta (Tool) do CrewAI
# ------------------------------------------------------------------
@tool("Ler últimos e-mails da Caixa de Entrada")
def fetch_recent_emails(max_results: int = 5) -> str:
    """Busca os e-mails mais recentes da caixa de entrada e retorna remetente, assunto e trecho."""
    try:
        service = get_gmail_service()
        results = service.users().messages().list(userId='me', maxResults=max_results, q="label:INBOX").execute()
        messages = results.get('messages', [])

        if not messages:
            return "Nenhum e-mail encontrado na caixa de entrada."

        email_list = []
        for msg in messages:
            txt = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            payload = txt.get('payload', {})
            headers = payload.get('headers', [])

            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'Sem Assunto')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Desconhecido')
            snippet = txt.get('snippet', '')

            email_list.append(f"De: {sender}\nAssunto: {subject}\nResumo inicial: {snippet}\n---")

        return "\n".join(email_list)
    except Exception as e:
        return f"Erro ao acessar o Gmail: {str(e)}"

# ------------------------------------------------------------------
# Configuração da LLM e Agentes
# ------------------------------------------------------------------
gemini_llm = LLM(
    model="gemini/gemini-3.6-flash",
    api_key=API_KEY,
)

email_fetcher_agent = Agent(
    role="Coletor e Triador de E-mails",
    goal="Acessar a caixa de entrada do usuário e extrair os e-mails mais recentes de forma estruturada.",
    backstory="Você é um assistente executivo encarregado de checar a caixa de entrada do usuário e compilar o que chegou de novo.",
    tools=[fetch_recent_emails],
    llm=gemini_llm,
    verbose=True
)

summary_agent = Agent(
    role="Sintetizador de Produtividade",
    goal="Transformar a lista de e-mails recebidos em um resumo diário claro, destacando urgências e pendências.",
    backstory="Você é um especialista em organização pessoal. Seu papel é filtrar o ruído e entregar um painel executivo direto ao ponto.",
    llm=gemini_llm,
    verbose=True
)

task_fetch = Task(
    description="Use a ferramenta de leitura de e-mails para buscar os últimos 5 e-mails da caixa de entrada do usuário.",
    expected_output="Uma lista crua contendo as informações dos e-mails coletados.",
    agent=email_fetcher_agent
)

task_summarize = Task(
    description="Analise os e-mails coletados na tarefa anterior e gere um resumo executivo em Português, "
                "curto e direto.\n"
                "Use SOMENTE esta formatação: '## ' para títulos, '**' para negrito e '- ' para itens. "
                "Sem tabelas, sem linhas '---', sem sub-itens.\n"
                "Cada e-mail deve ser UM único item em uma linha, no formato:\n"
                "- **Remetente**: assunto traduzido. Contexto em uma frase. **Ação:** o que fazer (só nos urgentes).\n\n"
                "Estrutura:\n"
                "## 🚨 Ações Urgentes/Importantes\n"
                "## 📬 Outros E-mails\n"
                "## 💡 Próximos Passos (no máximo 3 itens numerados no formato '- ')",
    expected_output="Resumo curto em Markdown simples, um item por e-mail.",
    agent=summary_agent
)

crew = Crew(
    agents=[email_fetcher_agent, summary_agent],
    tasks=[task_fetch, task_summarize],
    process=Process.sequential,
    verbose=True
)

# ------------------------------------------------------------------
# Modo bot: escuta comandos no Telegram (long polling)
# ------------------------------------------------------------------
def rodar_bot():
    base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    offset = None
    log.info("Bot iniciado na nuvem. Ouvindo comandos do Telegram...")

    # Notifica no Telegram que o bot subiu
    try:
        enviar_telegram("🟢 Bot iniciado na nuvem e ouvindo comandos!")
    except Exception:
        pass

    while True:
        try:
            r = requests.get(
                f"{base}/getUpdates",
                params={"timeout": 30, "offset": offset},
                timeout=40,
            )
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat_id = str(msg.get("chat", {}).get("id", ""))
                texto = (msg.get("text") or "").strip().lower()

                if chat_id != TELEGRAM_CHAT_ID:
                    continue

                log.info(f"Comando recebido: {texto}")

                if texto.startswith("/start"):
                    enviar_telegram(
                        "Oi! Envie /resumo para receber o resumo da sua caixa de entrada.\n"
                        "Bot rodando 24/7 na nuvem ☁️✅"
                    )
                elif texto.startswith("/resumo"):
                    enviar_telegram("⏳ Lendo sua caixa de entrada...")
                    try:
                        resultado = crew.kickoff()
                        enviar_telegram(str(resultado))
                        log.info("Resumo enviado com sucesso.")
                    except Exception as e:
                        log.error(f"Erro no kickoff: {e}\n{traceback.format_exc()}")
                        enviar_telegram(f"Erro ao gerar o resumo: {e}")
                elif texto.startswith("/status"):
                    uptime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    enviar_telegram(
                        f"✅ Bot online na nuvem (Railway)\n"
                        f"Hora do servidor: {uptime}"
                    )
        except requests.exceptions.RequestException as e:
            log.warning(f"Falha de rede: {e}. Tentando novamente em 5s...")
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Bot encerrado.")
            break
        except Exception as e:
            log.error(f"Erro inesperado no loop: {e}\n{traceback.format_exc()}")
            time.sleep(10)


# ------------------------------------------------------------------
# Execução
# ------------------------------------------------------------------
if __name__ == "__main__":
    log.info("=== Iniciando OpenClaw Bot (Cloud) ===")
    rodar_bot()
