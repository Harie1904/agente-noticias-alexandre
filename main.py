import os
import time
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

class TokenCounterCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens += token_usage.get('prompt_tokens', 0)
            self.completion_tokens += token_usage.get('completion_tokens', 0)
            self.total_tokens += token_usage.get('total_tokens', 0)
    
    def print_usage(self):
        print(f"\n{'='*50}") 
        print(f"TOKENS UTILIZADOS:")
        print(f"  Prompt tokens: {self.prompt_tokens}")
        print(f"  Completion tokens: {self.completion_tokens}")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"{'='*50}\n")

def search_news(query: str) -> str:
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return "ERRO: TAVILY_API_KEY não encontrada no arquivo .env"
        
        search = TavilySearchResults(
            api_key=tavily_api_key,
            max_results=5
        )
        results = search.invoke(query)
        
        if not results:
            return "Nenhuma notícia encontrada para este tópico."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Sem título')
            content = result.get('content', 'Sem conteúdo')
            url = result.get('url', '')
            formatted_results.append(f"{i}. {title}\n   {content}\n   Fonte: {url}")
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"ERRO ao buscar notícias: {str(e)}"

def analyze_sentiment(text: str, llm, token_counter) -> str:
    prompt = f"""Analise o sentimento do seguinte texto e classifique como POSITIVO ou NEGATIVO.
Forneça também uma justificativa curta (máximo 2 linhas) explicando o porquê da classificação.

Texto: {text}

Responda no seguinte formato:
Sentimento: [POSITIVO ou NEGATIVO]
Justificativa: [explicação curta]

Resposta:"""
    
    try:
        response = llm.invoke(prompt, config={"callbacks": [token_counter]})
        result = response.content.strip()
        return result
    except Exception as e:
        return f"ERRO ao analisar sentimento: {str(e)}"

def process_request(user_input: str, llm, token_counter):
    user_lower = user_input.lower()
    
    if "analise" in user_lower or "sentimento" in user_lower or "analisa" in user_lower:
        text_to_analyze = user_input
        for word in ["analise", "sentimento", "analisa", "o sentimento", "da notícia", "desta notícia", "do texto"]:
            text_to_analyze = text_to_analyze.replace(word, "")
        text_to_analyze = text_to_analyze.replace(":", "").strip()
        
        if len(text_to_analyze) < 20:
            return "Por favor, forneça o texto da notícia para análise. Exemplo: 'Analise o sentimento: [texto da notícia]'"
        
        sentiment_result = analyze_sentiment(text_to_analyze, llm, token_counter)
        return sentiment_result
    
    elif "busque" in user_lower or "buscar" in user_lower or "notícias" in user_lower or "noticias" in user_lower or "procure" in user_lower:
        results = search_news(user_input)
        
        if results.startswith("ERRO"):
            return results
        
        prompt = f"""Com base nos seguintes resultados de busca, crie um resumo MUITO CONCISO (máximo 3-5 tópicos curtos) das principais notícias:

Resultados: {results}

IMPORTANTE: Seja extremamente breve, máximo 3-5 bullet points!

Resumo:"""
        
        try:
            summary_response = llm.invoke(prompt, config={"callbacks": [token_counter]})
            summary = summary_response.content
            
            sentiment_prompt = f"""Analise o sentimento geral das seguintes notícias e classifique como POSITIVO ou NEGATIVO.
Forneça uma justificativa curta (máximo 1 linha).

Notícias: {summary}

Responda no formato:
Sentimento: [POSITIVO ou NEGATIVO]
Justificativa: [explicação curta em 1 linha]

Resposta:"""
            
            sentiment_response = llm.invoke(sentiment_prompt, config={"callbacks": [token_counter]})
            sentiment = sentiment_response.content.strip()
            
            return f"{summary}\n\n{'='*50}\nANÁLISE DE SENTIMENTO\n{'='*50}\n{sentiment}"
        except Exception as e:
            return f"ERRO ao processar resultados: {str(e)}"
    
    else:
        prompt = f"""Você é um assistente de notícias. Responda à seguinte pergunta de forma breve:

{user_input}

Resposta:"""
        
        try:
            response = llm.invoke(prompt, config={"callbacks": [token_counter]})
            return response.content
        except Exception as e:
            return f"ERRO: {str(e)}"

def main():
    print("="*50)
    print("ASSISTENTE AVALIADOR DE NOTÍCIAS")
    print("="*50)
    
    try:
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        
        if not mistral_api_key:
            raise ValueError("ERRO: MISTRAL_API_KEY não encontrada no arquivo .env")
        
        llm = ChatMistralAI(
            model="ministral-8b-latest",
            mistral_api_key=mistral_api_key,
            temperature=0.3
        )
        
        token_counter = TokenCounterCallback()
        
        print("\nAgente inicializado com sucesso!")
        print("Digite 'sair' para encerrar\n")
        print("Comandos disponíveis:")
        print("  - 'Busque notícias sobre [tópico]'")
        print("  - 'Analise o sentimento: [texto da notícia]'\n")
        
        while True:
            user_input = input("Você: ").strip()
            
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("\nEncerrando assistente...")
                token_counter.print_usage()
                break
            
            if not user_input:
                print("Por favor, digite uma pergunta válida.")
                continue
            
            try:
                response = process_request(user_input, llm, token_counter)
                print(f"\nAssistente: {response}\n")
            except Exception as e:
                print(f"\nERRO ao processar pergunta: {str(e)}\n")
    
    except Exception as e:
        print(f"\nERRO ao inicializar o agente: {str(e)}\n")
        return

if __name__ == "__main__":
    main()
